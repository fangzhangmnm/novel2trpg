import random
from transformers import TrainerCallback,GenerationConfig
from dataclasses import dataclass, field



class TextGenerationCallback(TrainerCallback):
    def __init__(self,interval,max_new_tokens,model,tokenizer,train_dataset,logger):
        self.interval=interval
        self.model=model
        self.tokenizer=tokenizer
        self.train_dataset=train_dataset
        self.generate_config=GenerationConfig(max_new_tokens=max_new_tokens)
        self.logger=logger

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval == 0 and self.interval != -1:
            # try:
            i=random.randint(0,len(self.train_dataset['input_ids']))
            input_ids=self.train_dataset['input_ids'][i]
            input_text=self.tokenizer.decode(input_ids)
            label_ids=self.train_dataset['labels'][i]
            label_text=self.tokenizer.decode(label_ids)
            input_text1=input_text[:-len(label_text)]

            input_ids = self.tokenizer.encode(input_text1, return_tensors='pt').to(self.model.device)
            output_ids = self.model.generate(input_ids, do_sample=True, max_new_tokens=50,generation_config=self.generate_config)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            output_text1=output_text[len(input_text1):]
            input_text1=input_text1.replace('\n','\\n')
            output_text1=output_text1.replace('\n','\\n')
            label_text=label_text.replace('\n','\\n')
            self.logger.info(f"input: {input_text1}")
            self.logger.info(f"prediction: {output_text1}")
            self.logger.info(f"ground_truth: {label_text}")
            # breakpoint()
            # except Exception as e:
            #     print(e)
            #     breakpoint()

@dataclass
class TextGenerationCallbackArguments:
    """
    Arguments pertaining to the text generation callback.
    """
    preview_text_generation_interval: int = field(
        default=-1,
        metadata={"help": "The interval for text generation, -1 means no text generation"}
    )