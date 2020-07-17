# 基于pytorch的中文语言模型预训练
ACL2020 Best Paper有一篇论文提名奖，《Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks》。这篇论文做了很多语言模型预训练的实验，系统的分析了语言模型预训练对子任务的效果提升情况。有几个主要结论：
* 在目标领域的数据集上继续预训练（DAPT）可以提升效果；目标领域的语料与RoBERTa的原始预训练语料越不相关，DAPT效果则提升更明显。

* 在具体任务的数据集上继续预训练（TAPT）可以十分“廉价”地提升效果。

* 结合二者（先进行DAPT，再进行TAPT）可以进一步提升效果。

* 如果能获取更多的、任务相关的无标注数据继续预训练（Curated-TAPT），效果则最佳。

* 如果无法获取更多的、任务相关的无标注数据，采取一种十分轻量化的简单数据选择策略，效果也会提升。



虽然在bert上语言模型预训练在算法比赛中已经是一个稳定的上分操作。但是上面这篇文章难能可贵的是对这个操作进行了系统分析。大部分中文语言模型都是在tensorflow上训练的，一个常见例子是中文roberta项目。可以参考
https://github.com/brightmart/roberta_zh


使用pytorch进行中文bert语言模型预训练的例子比较少。在huggingface的Transformers中，有一部分代码支持语言模型预训练(不是很丰富，很多功能都不支持比如wwm)。为了用最少的代码成本完成bert语言模型预训练，本文借鉴了里面的一些现成代码。也尝试分享一下使用pytorch进行语言模型预训练的一些经验。主要有三个常见的中文语言模型
* bert-base-chinese
* roberta-wwm-ext
* ernie

## bert-base-chinese

(https://huggingface.co/bert-base-chinese)
​

这是最常见的中文bert语言模型，基于中文维基百科相关语料进行预训练。把它作为baseline，在领域内无监督数据进行语言模型预训练很简单。只需要使用官方给的例子就好。

https://github.com/huggingface/transformers/tree/master/examples/language-modeling
(本文使用的transformers更新到3.0.2)
```
python run_language_model_bert.py     --output_dir=output     --model_type=bert     --model_name_or_path=bert-base-chinese     --do_train     --train_data_file=train.txt     --do_eval     --eval_data_file=eval.txt     --mlm --per_device_train_batch_size=4

```

## roberta-wwm-ext

(https://github.com/ymcui/Chinese-BERT-wwm)


哈工大讯飞联合实验室发布的预训练语言模型。预训练的方式是采用roberta类似的方法，比如动态mask，更多的训练数据等等。在很多任务中，该模型效果要优于bert-base-chinese。
因为中文roberta类的配置文件比如vocab.txt，都是采用bert的方法设计的。英文roberta模型读取配置文件的格式默认是vocab.json。对于一些英文roberta模型，倒是可以通过AutoModel自动读取。这就解释了huggingface的模型库的中文roberta示例代码为什么跑不通。https://huggingface.co/models?


如果要基于上面的代码run_language_modeling.py继续预训练roberta。还需要做两个改动。
* 下载roberta-wwm-ext到本地目录hflroberta，在config.json中修改“model_type”:"roberta"为"model_type":"bert"。
* 对上面的run_language_modeling.py中的AutoModel和AutoTokenizer都进行替换为BertModel和BertTokenizer。

假设config.json已经改好，可以运行如下命令。
```
python run_language_model_roberta.py     --output_dir=output     --model_type=bert     --model_name_or_path=hflroberta     --do_train     --train_data_file=train.txt     --do_eval     --eval_data_file=eval.txt     --mlm --per_device_train_batch_size=4
```

### ernie
https://github.com/nghuyong/ERNIE-Pytorch）

ernie是百度发布的基于百度知道贴吧等中文语料结合实体预测等任务生成的预训练模型。这个模型的准确率在某些任务上要优于bert-base-chinese和roberta。如果基于ernie1.0模型做领域数据预训练的话只需要一步修改。

* 下载ernie1.0到本地目录ernie，在config.json中增加字段"model_type":"bert"。
运行
```
python run_language_model_ernie.py     --output_dir=output     --model_type=bert     --model_name_or_path=ernie     --do_train     --train_data_file=train.txt     --do_eval     --eval_data_file=eval.txt     --mlm --per_device_train_batch_size=4

```
