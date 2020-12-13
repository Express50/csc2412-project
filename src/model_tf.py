# from torch import Tensor
# from torch.nn.modules.activation import ReLU
# from torch.nn.modules.dropout import Dropout
# from torch.nn.modules.linear import Linear
# from torch.nn.modules.module import Module
from tensorflow import Tensor
from tensorflow.keras.activations import relu as ReLU, sigmoid
from tensorflow.keras.layers import Dropout, Dense as Linear
from tensorflow.keras import Model
from transformers import AutoConfig, TFAutoModel

import inspect
from transformers.modeling_distilbert import DistilBertModel

from transformers.modeling_gpt2 import GPT2Model
from transformers.modeling_t5 import T5Model

def get_args(func):
    sig = inspect.signature(func)
    kwargs = []

    for param in sig.parameters.values():
        if param.kind != param.VAR_KEYWORD:
            kwargs.append(param.name)

    return kwargs

# TODO: may need to freeze all but top layers of transformer for lower mem
class SentimentAnalysisModel(Model):
    def __init__(self, model_name: str, output_dim: int) -> None:
        super(SentimentAnalysisModel, self).__init__()

        config = AutoConfig.from_pretrained(model_name)

        self.transformer = TFAutoModel.from_pretrained(model_name)

        # freeze all but last layer of transformer
        # layers_to_freeze = None
        # frozen_params = 0
        # if type(self.transformer) is GPT2Model:
        #     layers_to_freeze = self.transformer.h[:-1]
        # elif type(self.transformer) is DistilBertModel:
        #     layers_to_freeze = self.transformer.transformer.layer[:-1]
        # elif type(self.transformer) is T5Model:
        #     layers_to_freeze = self.transformer.encoder.block[:-1]
        #     layers_to_freeze.extend(self.transformer.decoder.block[:-1])

        # for layer in layers_to_freeze:
        #     for param in layer.parameters():
        #         param.requires_grad = False
        #         frozen_params += param.numel()

        # print(f'Init model: frozen {frozen_params} params.')

        self.pre_classifier = Linear(units=config.hidden_size, input_dim=config.hidden_size, activation='linear')
        self.dropout = Dropout(0.3)
        # self.classifier = Linear(units=output_dim, input_dim=config.hidden_size, activation='linear')
        self.classifier = Linear(units=1, input_dim=config.hidden_size, activation='linear')

    def call(self, args): #input_ids, attention_mask, token_type_ids) -> Tensor:
        # Some models don't require token_type_ids, so only pass supported args to forward()
        # print(args)

        # all_args = {
        #     'input_ids': args["input_ids"],
        #     'attention_mask': args["attention_mask"],
        #     'token_type_ids': args["token_type_ids"]
        # }

        # supported_args = get_args(self.transformer.call)
        # args = {}
        # for arg in supported_args:
        #     if arg in all_args:
        #         args[arg] = all_args[arg]

        print(args)
        input_ids = args.pop('input_ids')

        args.pop('token_type_ids')


        transformer_output = self.transformer(input_ids, **args)

        hidden_state = transformer_output[0]

        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = ReLU(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        return output


if __name__ == "__main__":
    model = SentimentAnalysisModel('distilgpt2', 5)

    print(type(model.transformer))
