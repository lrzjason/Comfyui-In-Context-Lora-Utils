# Comfyui-In-Context-Lora-Utils

## Installation
1. Download the zip file containing the custom nodes.
2. Extract the files to the following directory:  
   `..\ComfyUI\custom_nodes`
3. Restart **ComfyUI** to load the new nodes.


## Workflow
A complete workflow demonstrating the usage of this extension is available here:  
[ComfyUI-In-Context-LoRA Workflow](https://civitai.com/models/933018?modelVersionId=1131311)


## Prerequisites
To make the workflow functional, ensure the following files are downloaded and placed in the appropriate directories:
1. **`flux1-fill-dev.safetensors`** [Download here](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/resolve/main/flux1-fill-dev.safetensors?download=true)  
   Place in:  `..\ComfyUI\ComfyUI\models\unet`
   
3. **`ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors`** [Download here](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14/resolve/main/ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors)  
   Place in: `..\ComfyUI\ComfyUI\models\clip`
   
5. **`t5xxl_fp8_e4m3fn_scaled.safetensors`** [Download here](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors)  
   Place in: `..\ComfyUI\ComfyUI\models\clip`
   
7. **`ae.sft`** [Download here](https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors)  
   Place in: `..\ComfyUI\models\vae`
   
9. **`flux1-redux-dev.safetensors`** [Download here](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors?download=true)  
   Place in: `..\ComfyUI\models\style_models`
   
11. **`sigclip_vision_patch14_384.safetensors`** [Download here](https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors)  
   Place in: `..\ComfyUI\models\clip_vision`


## Change Logs:
- **2024-12-14:** Adjust x_diff calculation and adjust fit image logic.
- **2024-12-13:** Fix Incorrect Padding
- **2024-12-12(2):** Fix center point calculation when close to edge.
- **2024-12-12:** Reconstruct the node with new caculation.
- **2024-12-11:** Avoid too large buffer cause incorrect context area
- **2024-12-10(3):** Avoid padding when image have width or height to extend the context area
- **2024-12-05:** Fix incorrect cropping issue when mask is not fit in target ratio
- **2024-12-01:** Adjust contours calculation, fix multiple contour only produce single contour issue
- **2024-11-30:** Add AutoPatch node, it is able to automatically select patch mode and patch type
- **2024-11-29:** Recontruct the node and seperate from old node, new nodes: CreateContextWindow, ConcatContextWindow
- **2024-11-22:** Update Two Images input and related masks input


## Example:  
- v3 Object Replacement
![alt text](https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/v3_object_replacement.png)
- v3 Generate On Target Position
![alt text](https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/v3_target_position.png)
- v3 Virtual Try On
![alt text](https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/v3_try_on.png)


## OLD NODE Example:
Simple Try On Lora:
https://civitai.com/models/950111/flux-simple-try-on-in-context-lora

![alt text](https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/example_1.png)
![alt text](https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/example_2.png)


## Contact
- **Twitter**: [@Lrzjason](https://twitter.com/Lrzjason)  
- **Email**: lrzjason@gmail.com  
- **QQ Group**: 866612947  
- **Civitai**: [xiaozhijason](https://civitai.com/user/xiaozhijason)


## Sponsors me for more open source projects:
<div align="center">
  <table>
    <tr>
      <td align="center">
        <p>Buy me a coffee:</p>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/bmc_qr.png" alt="Buy Me a Coffee QR" width="200" />
      </td>
      <td align="center">
        <p>WeChat:</p>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/wechat.jpg" alt="WeChat QR" width="200" />
      </td>
    </tr>
  </table>
</div>
