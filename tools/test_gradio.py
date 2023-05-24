import sys


sys.argv = ["tools/vits_gradio.py",
            "--model_path", "/Users/peng/PROGRAM/GitHub/so-vits-svc/lain/G_256800_infer.pth",
            "--config_path", "/Users/peng/PROGRAM/GitHub/so-vits-svc/lain/config.json",
            "--cluster_model_path", "/Users/peng/PROGRAM/GitHub/so-vits-svc/logs/lain/kmeans_10000.pt",
            # "--debug",
            ]

from tools import lain_gradio
lain_gradio.main()
