import re

from typing_extensions import override

import comfy.samplers
from comfy_api.latest import ComfyExtension, io


class MOSSImagePAG(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MOSSImagePAG",
            display_name="MOSS-Image Perturbed Attention Guidance",
            category="model/patch",
            description="Perturbed attention guidance as in diffusers: an extra conditional pass where the self-attention of the chosen blocks returns its values, then cfg + scale * (cond - perturbed).",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("scale", default=3.0, min=0.0, max=100.0, step=0.01, tooltip="PAG scale. 0 disables the extra pass."),
                io.String.Input("layers", default="20", tooltip="Indices of the transformer blocks to perturb, e.g. \"18, 20, 22\" (MOSS-Image has 42)."),
                io.Float.Input("adaptive_scale", default=0.0, min=0.0, max=100.0, step=0.001, tooltip="Diffusers' pag_adaptive_scale: the scale is reduced by adaptive_scale * (1000 - t), so PAG fades in as the noise level drops."),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, scale, layers, adaptive_scale) -> io.NodeOutput:
        blocks = {int(i) for i in re.findall(r"\d+", layers)}
        m = model.clone()

        def post_cfg_function(args):
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            pag_scale = scale - adaptive_scale * (1000 - sigma.max().item() * 1000)
            if pag_scale <= 0:
                return cfg_result

            model_options = args["model_options"]
            transformer_options = model_options.get("transformer_options", {})
            previous = transformer_options.get("optimized_attention_override")

            def perturbed_attention(func, q, k, v, *rest, **kwargs):
                if kwargs["transformer_options"].get("block_index") in blocks:
                    # k and v also hold the cached reference tokens, which have no query
                    return v[:, :q.shape[1]]
                return (previous or (lambda f, *a, **kw: f(*a, **kw)))(func, q, k, v, *rest, **kwargs)

            model_options = {**model_options, "transformer_options": {**transformer_options, "optimized_attention_override": perturbed_attention}}
            (perturbed,) = comfy.samplers.calc_cond_batch(args["model"], [args["cond"]], args["input"], sigma, model_options)
            return cfg_result + (args["cond_denoised"] - perturbed) * pag_scale

        m.set_model_sampler_post_cfg_function(post_cfg_function)
        return io.NodeOutput(m)


class MOSSImagePAGExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MOSSImagePAG]


async def comfy_entrypoint() -> MOSSImagePAGExtension:
    return MOSSImagePAGExtension()
