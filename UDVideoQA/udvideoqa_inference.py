import sys
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Unified Inference script for finetuned models.")
    parser.add_argument("--model", required=True, choices=["intern_vl3","llava-next-video","qwen-vl2-7b"], help="Model type for inference.")
    parser.add_argument("--model_path", required=True, help="Path to the finetuned model.")
    parser.add_argument("--annotations_path", required=True, help="Path to the annotations file.")
    parser.add_argument("--videos_dir", required=True, help="Directory containing the videos.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output answers.")
    args = parser.parse_args()

    if args.model == "intern_vl3":
        from InterVL3.scripts import inference_intern_vl3
        inference_intern_vl3(
            model_path=args.model_path,
            annotations_path=args.annotations_path,
            videos_dir=args.videos_dir,
            output_dir=args.output_dir
        )
    elif args.model == "llava-next-video":  
        from LlavaNextVideo import inference_llava_next_video
        inference_llava_next_video(
            model_path=args.model_path,
            annotations_path=args.annotations_path,
            videos_dir=args.videos_dir,
            output_dir=args.output_dir
        )
    elif args.model == "qwen-vl2-7b":
        from QwenVL2.scripts import inference_qwen_vl2
        inference_qwen_vl2(
            model_path=args.model_path,
            annotations_path=args.annotations_path,
            videos_dir=args.videos_dir,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()