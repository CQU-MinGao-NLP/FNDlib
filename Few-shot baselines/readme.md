少样本虚假新闻检测baselines

* Bert-Prototypical Networks：是一种基于度量的方法，通过使用样本平均值作为类原型来进行少样本分类，使用bert做特征提取器。

* DAFD：是一个预训练微调的少样本虚假新闻检测框架，它使用HAN做文本特征提取器，在预训练的过程中应用领域适配技术对齐目标域和源域的数据分布，在微调的过程中使用对抗学习增加模型的鲁棒性和泛化能力。
