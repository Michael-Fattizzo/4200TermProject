from __future__ import annotations

import argparse
import threading # Added for GUI
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import torch

import ScreenCapture as sc
from ShogiUI import ShogiControlGUI # Added UI Import
from converter import encode_position
from moveEncoding import TOTAL_MOVE_CLASSES, encode_move_obj
from shogiEngine import (
    BLACK,
    WHITE,
    Move,
    Position,
    apply_move,
    generate_legal_moves,
    opponent,
    position_from_capture_state,
    suggest_move_from_capture_state,
    square_to_usi,
)
from train_shogi import ShogiPolicyValueNet


MODEL_MODE = "model"
BASIC_MODE = "basic"

# --- GUI BRIDGE ---
remote_key = -1
gui_instance = None # Added to allow main() to find the window

def handle_gui_button(key_char):
    global remote_key
    remote_key = ord(key_char)


def parse_side(value: str) -> str:
    value = value.strip().lower()
    if value in {"black", "sente", "b"}:
        return BLACK
    if value in {"white", "gote", "w"}:
        return WHITE
    raise argparse.ArgumentTypeError("side must be black/sente/b or white/gote/w")


def format_move(move: Move) -> str:
    if move.drop:
        return f"{move.usi()}  | drop {move.piece} on {square_to_usi(move.to_sq)}"

    text = f"{move.usi()}  | move {move.piece} from {square_to_usi(move.from_sq)} to {square_to_usi(move.to_sq)}"
    if move.promote:
        text += " and promote"
    if move.captured:
        text += f"; captures {move.captured}"
    return text


class NeuralMoveSuggester:
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.model = ShogiPolicyValueNet(
            in_channels=ckpt.get("channels", 44),
            num_policy_classes=ckpt.get("num_policy_classes", TOTAL_MOVE_CLASSES),
            width=ckpt.get("width", 128),
            blocks=ckpt.get("blocks", 6),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def suggest(self, capture_state: Dict[str, object], side_to_move: str, ply: int = 1) -> Optional[Dict[str, object]]:
        position = position_from_capture_state(capture_state, side_to_move=side_to_move)
        legal_moves = generate_legal_moves(position, side_to_move)
        if not legal_moves:
            return None

        x = torch.tensor(encode_position(position, ply_index=ply), dtype=torch.float32)
        x = x.unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(x)
            logits = policy_logits[0]

        best_move = max(legal_moves, key=lambda mv: float(logits[encode_move_obj(mv)].item()))
        best_id = encode_move_obj(best_move)
        prob = torch.softmax(logits[[encode_move_obj(mv) for mv in legal_moves]], dim=0)
        legal_ids = [encode_move_obj(mv) for mv in legal_moves]
        best_prob = float(prob[legal_ids.index(best_id)].item())

        return {
            "move": best_move.usi(),
            "move_obj": best_move,
            "score": float(logits[best_id].item()),
            "legal_policy_prob": best_prob,
            "value": float(value.item()),
            "explanation": format_move(best_move),
        }


def print_help() -> None:
    print("\nControls")
    print("  q = quit")
    print("  r = re-detect board")
    print("  s = save calibration crops")
    print("  d = save all 81 board cells")
    print("  a = show 81-cell grid")
    print("  b = use basic engine")
    print("  m = use trained AI model")
    print("  t = toggle side to move")
    print("  h = show this help")


def choose_mode(args: argparse.Namespace) -> str:
    if args.engine in {BASIC_MODE, MODEL_MODE}:
        return args.engine

    print("Choose move suggestion engine:")
    print("  1 = trained AI model")
    print("  2 = basic search engine")
    choice = input("Enter 1 or 2: ").strip()
    return MODEL_MODE if choice == "1" else BASIC_MODE


def load_neural_if_needed(mode: str, model_path: Optional[str]) -> Optional[NeuralMoveSuggester]:
    if mode != MODEL_MODE:
        return None
    if not model_path:
        print("No --model checkpoint was provided. Falling back to the basic engine.")
        return None
    if not Path(model_path).exists():
        print(f"Model checkpoint not found: {model_path}. Falling back to the basic engine.")
        return None
    return NeuralMoveSuggester(model_path)


def main() -> None:
    global remote_key, gui_instance
    parser = argparse.ArgumentParser(
        description="Run the shogi screen reader and optionally suggest moves with a trained AI model or the basic engine."
    )
    parser.add_argument("--engine", choices=["ask", BASIC_MODE, MODEL_MODE], default="ask")
    parser.add_argument("--model", default="shogi_policy_value.pt", help="Path to a train_shogi.py checkpoint.")
    parser.add_argument("--side", type=parse_side, default=BLACK, help="Side to move: black/sente/b or white/gote/w.")
    parser.add_argument("--basic-depth", type=int, default=2, help="Search depth for the basic engine.")
    parser.add_argument("--suggest-every", type=float, default=2.0, help="Seconds between printed suggestions.")
    parser.add_argument("--no-windows", action="store_true", help="Do not open OpenCV preview windows.")
    args = parser.parse_args()

    print("Starting shogi screen reader...")
    print_help()

    mode = choose_mode(args)
    neural = load_neural_if_needed(mode, args.model)
    if mode == MODEL_MODE and neural is None:
        mode = BASIC_MODE

    occupancy_classifier = sc.OccupancyCNNClassifier.load(sc.OCCUPANCY_CHECKPOINT)
    piece_classifier = sc.BoardCNNClassifier.load(sc.CNN_CHECKPOINT)
    templates = sc.load_templates(sc.TEMPLATE_ROOT)
    digit_templates = sc.load_digit_templates(sc.TEMPLATE_ROOT)

    screen, regions, active_monitor = sc.detect_all_regions()
    print("\nDetected regions:")
    print("board     =", regions.board.as_dict())
    print("left_hand =", regions.left_hand.as_dict())
    print("right_hand=", regions.right_hand.as_dict())
    print("monitor   =", active_monitor)
    print(f"Current engine: {mode}")
    print(f"Side to move: {args.side}")

    last_suggestion_time = 0.0
    ply = 1

    while True:
        screen = sc.capture_monitor(active_monitor)
        state = sc.extract_state(
            screen,
            regions,
            occupancy_classifier,
            piece_classifier,
            templates,
            digit_templates,
        )

        now = cv2.getTickCount() / cv2.getTickFrequency()
        if now - last_suggestion_time >= args.suggest_every:
            try:
                if mode == MODEL_MODE and neural is not None:
                    suggestion = neural.suggest(state, side_to_move=args.side, ply=ply)
                    if suggestion is None:
                        print(f"\n[{mode}] No legal move found for {args.side}.")
                    else:
                        print(
                            f"\n[{mode}] {args.side} to move: {suggestion['explanation']} "
                            f"| legal-policy={suggestion['legal_policy_prob']:.3f} "
                            f"| value={suggestion['value']:.3f}"
                        )
                        # UPDATE GUI
                        if gui_instance:
                            gui_instance.update_display(args.side, suggestion['move'], suggestion['explanation'], suggestion['value'], suggestion['legal_policy_prob'])
                else:
                    suggestion = suggest_move_from_capture_state(
                        state,
                        side_to_move=args.side,
                        depth=args.basic_depth,
                    )
                    if suggestion is None:
                        print(f"\n[{mode}] No legal move found for {args.side}.")
                    else:
                        print(f"\n[{mode}] {args.side} to move: {suggestion['move']} | {suggestion['explanation']}")
                        # UPDATE GUI
                        if gui_instance:
                            gui_instance.update_display(args.side, suggestion['move'], suggestion['explanation'])
            except Exception as exc:
                print(f"\nCould not produce suggestion: {exc}")

            last_suggestion_time = now

        if not args.no_windows:
            overlay = sc.draw_regions(screen, regions)
            board_img = sc.crop_region(screen, regions.board)
            left_hand_img = sc.crop_region(screen, regions.left_hand)
            right_hand_img = sc.crop_region(screen, regions.right_hand)

            cv2.imshow("Shogi Capture - Overlay", overlay)
            cv2.imshow("Shogi Capture - Board", sc.draw_board_grid(board_img))
            cv2.imshow("Shogi Capture - Left Hand", sc.draw_hand_slots(left_hand_img))
            cv2.imshow("Shogi Capture - Right Hand", sc.draw_hand_slots(right_hand_img))

        # Handle Inputs (GUI or Keyboard)
        if remote_key != -1:
            key = remote_key
            remote_key = -1
        else:
            key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("h"):
            print_help()
        elif key == ord("b"):
            mode = BASIC_MODE
            print("Using basic engine.")
        elif key == ord("m"):
            if neural is None:
                neural = load_neural_if_needed(MODEL_MODE, args.model)
            if neural is not None:
                mode = MODEL_MODE
                print("Using trained AI model.")
        elif key == ord("t"):
            args.side = opponent(args.side)
            ply += 1
            print(f"Side to move: {args.side}")
            if gui_instance:
                gui_instance.update_display(args.side, "--", "Side toggled.")
        elif key == ord("r"):
            screen, regions, active_monitor = sc.detect_all_regions()
            print("\nRe-detected regions:")
            print("board     =", regions.board.as_dict())
            print("left_hand =", regions.left_hand.as_dict())
            print("right_hand=", regions.right_hand.as_dict())
            print("monitor   =", active_monitor)
        elif key == ord("s"):
            sc.save_calibration_images(screen, regions)
            print(f"Saved debug images to ./{sc.SAVE_DIR}/")
        elif key == ord("d"):
            board_img = sc.crop_region(screen, regions.board)
            saved_dir = sc.save_board_cells(board_img)
            print(f"Saved 81 board cells to ./{saved_dir}/")
        elif key == ord("a"):
            board_img = sc.crop_region(screen, regions.board)
            sc.show_all_cells(board_img)
            print("Showing 81-cell grid preview.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # START MAIN IN A THREAD, START GUI ON THE MAIN LINE
    gui_instance = ShogiControlGUI(handle_gui_button)
    threading.Thread(target=main, daemon=True).start()
    gui_instance.mainloop()