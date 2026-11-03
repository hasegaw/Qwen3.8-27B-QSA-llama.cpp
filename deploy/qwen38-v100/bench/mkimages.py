# test images with known content (the number 7391, a red circle, a blue rectangle, a green triangle) for imgtest.py
import os, glob, sys
from PIL import Image, ImageDraw, ImageFont
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "testimg")
os.makedirs(out, exist_ok=True)
fonts = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
for name, W, H in (("small.png", 640, 360), ("large.png", 1920, 1080), ("4k.png", 3840, 2160), ("8k.png", 7680, 4320)):
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    f = ImageFont.truetype(fonts[0], H // 6) if fonts else ImageFont.load_default()
    d.text((W * 0.08, H * 0.1), "7391", fill="black", font=f)
    d.ellipse([W*0.55, H*0.15, W*0.75, H*0.5], fill="red"); d.rectangle([W*0.1, H*0.6, W*0.35, H*0.9], fill="blue")
    d.polygon([(W*0.6, H*0.9), (W*0.8, H*0.6), (W*0.95, H*0.9)], fill="green"); im.save(os.path.join(out, name))
print("wrote", out)
