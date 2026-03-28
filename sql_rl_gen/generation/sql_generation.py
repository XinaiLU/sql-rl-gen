import logging
import os
import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, HfArgumentParser
from configs.config import SPIDER_DATABASES_PATH, WIKISQL_PATH, BIRD_DATABASES_DEV_PATH
from configs.data_args import DataArguments, DatasetName
from configs.rl_args import TrainingArguments
from data_preprocess.data_utils import get_dataset
from sql_rl_gen.generation.envs.sql_generation_environment import SQLRLEnv
from sql_rl_gen.generation.envs.utils import prepare_observation_list_and_dataset_to_pass, find_device, count_parameters
from sql_rl_gen.generation.rllib.custom_actor import CustomActor
from sql_rl_gen.generation.rllib.custom_trainer import train_evaulate_agent

def train_llm(train_args: TrainingArguments, data_args: DataArguments):

    dataset = get_dataset(data_args)
    dataset = dataset.take(train_args.number_of_rows_to_use)

    if data_args.dataset_name == DatasetName.SPIDER.value:
        dataset_path = SPIDER_DATABASES_PATH
    elif data_args.dataset_name == DatasetName.WIKISQL.value:
        dataset_path = WIKISQL_PATH
    else:
        dataset_path = BIRD_DATABASES_DEV_PATH

    # obseravtion 每一步强化学习里，agent 看到的“当前状态输入”
    # 在这个项目中，是用户问题、数据schema、可能还有提示模板
    # 本质上是一堆可以喂给 LLM 生成 SQL 的输入样本
    observation_list, data_list_to_pass, columns_names_mismatch = prepare_observation_list_and_dataset_to_pass(dataset)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
    logger = logging.getLogger("Train")
    os.makedirs(train_args.outdir, exist_ok=True)

    # model + tokenizer
    # 这里加载的是生成SQL的主模型
    # 输入：问题 + schema
    # 输出：SQL 语句
    tokenizer = AutoTokenizer.from_pretrained(train_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(train_args.model_name_or_path)
    model.train()

    tokenizer.sep_token = ';'
    device = find_device()
    model = model.to(device)
    trainable_params, all_param = count_parameters(model)
    logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(trainable_params, all_param, 100 * trainable_params / all_param))

    # 强化学习里的环境（environment）
    # 给出当前 observation, 接收 agent 生成的 action, 执行 action, 返回 reward、done、next observation
    env = SQLRLEnv(model, tokenizer, data_list_to_pass, dataset_path, 
                    train_args.outdir, logger, "training", 
                    columns_names_mismatch=columns_names_mismatch,
                    observation_input=observation_list, 
                    compare_sample=1)
    
    # 把 observation 送进 tokenizer
    # 调用模型生成 token
    # 按 temperature / top_k / top_p 采样
    # 输出 action（也就是 SQL）
    actor = CustomActor(env, model, tokenizer, temperature=train_args.temperature, top_k=train_args.top_k, top_p=train_args.top_p)

    # 负责“怎么根据 reward 学习更新策略”
    agent = actor.agent_ppo(update_interval=train_args.update_interval, minibatch_size=train_args.minibatch_size, epochs=train_args.epochs, lr=train_args.lr)
    print(actor.predict(observation_list[35]))

    # 训练过程
    logger.info("------------TRAINING IS STARTED------------")
    train_evaulate_agent(agent, env, steps=train_args.steps_n, eval_n_steps=None, eval_n_episodes=train_args.eval_n_episodes, train_max_episode_len=train_args.train_max_episode_len,
                         eval_interval=train_args.eval_interval, outdir=f'{train_args.outdir}', logger=logger)
    logger.info("------------TRAINING IS FINISHED------------")

def run():
    parser = HfArgumentParser((TrainingArguments, DataArguments))
    train_args, data_args = parser.parse_args_into_dataclasses()
    data_args.init_for_training()
    train_llm(train_args, data_args)

if __name__ == "__main__":
    run()