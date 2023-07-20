import argparse
import datetime
import ast
import os

import cv2
import torch

from utils.tools import convert_box_xywh_to_xyxy
from fastsam import FastSAM, FastSAMPrompt, FastSAMPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM-s.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="0", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/test_pro", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str,
        default='[[0, 0]]',
        help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument(
        "--box_prompt", type=str, default="[[0,0,0,0]]",
        help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()


def main(args):
    # load model
    model = FastSAM(args.model_path)
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)

    if args.img_path.isnumeric():
        cap = cv2.VideoCapture(int(args.img_path))
    else:
        cap = cv2.VideoCapture(args.img_path)

    if args.output is not None:
        save_dir = os.path.join(
            args.output, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        output = cv2.VideoWriter(os.path.join(save_dir, 'result.mp4'), fourcc, fps, size)

    overrides = model.overrides.copy()
    overrides['conf'] = 0.25
    overrides.update(device=args.device, retina_masks=args.retina, imgsz=args.imgsz, conf=args.conf,
                     iou=args.iou)  # prefer kwargs
    overrides['mode'] = 'predict'
    assert overrides['mode'] in ['track', 'predict']
    overrides['save'] = False
    model.predictor = FastSAMPredictor(overrides=overrides)
    model.predictor.setup_model(model=model.model, verbose=False)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            everything_results = model.predictor(frame, stream=False)

            bboxes = None
            points = None
            point_label = None
            prompt_process = FastSAMPrompt(False, everything_results, device=args.device)
            prompt_process.ori_img = frame
            if args.box_prompt[0][2] != 0 and args.box_prompt[0][3] != 0:
                ann = prompt_process.box_prompt(bboxes=args.box_prompt)
                bboxes = args.box_prompt
            elif args.text_prompt != None:
                ann = prompt_process.text_prompt(text=args.text_prompt)
            elif args.point_prompt[0] != [0, 0]:
                ann = prompt_process.point_prompt(
                    points=args.point_prompt, pointlabel=args.point_label
                )
                points = args.point_prompt
                point_label = args.point_label
            else:
                ann = prompt_process.everything_prompt()

            annotated_frame = prompt_process.plot_to_result(
                annotations=ann,
                bboxes=bboxes,
                points=points,
                point_label=point_label,
                withContours=args.withContours,
                better_quality=args.better_quality,
            )

            if args.output is not None:
                output.write(annotated_frame)
            cv2.imshow("frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Release capture and destroy windows at the end of the video
    cv2.destroyAllWindows()
    cap.release()
    if args.output is not None:
        output.release()


if __name__ == "__main__":
    args = parse_args()
    main(args)
