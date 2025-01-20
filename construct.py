import os
os.environ["REPLICATE_API_TOKEN"]="r8_ZKRbZEbx2j1K3Kgxw5SVqMuGnPwmEcT0NTN1X"

import base64
from imageio import imread, imwrite
import os
import replicate

def process_images(original_image_path, processed_image_path, output_dir):
    with open(original_image_path, 'rb') as file:
        data = base64.b64encode(file.read()).decode('utf-8')
        original_image = f"data:application/octet-stream;base64,{data}"

    with open(processed_image_path, 'rb') as file:
        data = base64.b64encode(file.read()).decode('utf-8')
        processed_image = f"data:application/octet-stream;base64,{data}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_inference_steps_list = [10]

    for i, num_inference_steps in enumerate(num_inference_steps_list):
        output = replicate.run(
            "andreasjansson/stable-diffusion-inpainting:e490d072a34a94a11e9711ed5a6ba621c3fab884eda1665d9d3a282d65a21180",
            input={
                "mask": processed_image,
                "image": original_image,
                "prompt": "reconstruct the statue",
                "invert_mask": False,
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "negative_prompt": "",
                "num_inference_steps": num_inference_steps
            }
        )
        output_path = os.path.join(output_dir, f"output_{i+1}.png")
        imwrite(output_path, imread(output[0]))


