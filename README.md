# 基于ChatGLM-6B的动漫角色调校实验

Training a Large Language Model to speak like an anime character using p-tuning

使用[ChatGLM](https://github.com/THUDM/ChatGLM-6B/)默认自带的ptuning脚本，让语言模型学习动漫人物的语言风格和小说中的设定。训练数据又使用ChatGPT蒸馏小说得到。

# 使用方法

STEP1_build_knowledge_base.ipynb 构建背景知识库

```
Query: 
才人：起床了！\n路易斯：
LookUp:
【露易丝 的性格易怒，容易骂人，有点糊涂（跟才人比），为了小事常常生气。（0.17）】
【拉·瓦利埃尔公爵 在托里斯特因有很大的影响力，涉足军、政、商三界，与王室关系密切，动画版第十二话认可了露易丝与才人的关系，同意让露易丝嫁给才人，在罗马尼亚大圣堂做了见证。（0.17）】
【平贺才人 已与露易丝同居中。同时被书中许多女性角色爱慕著。喜欢露易丝。（0.14）】
【平贺才人 他的称号：修瓦里埃骑士（见习骑士），子爵(结尾被封) 他喜欢的人：露易丝（表白过很多次） 他的妻子:露易丝（在大结局和露易丝结婚）（0.12）】
【平贺才人 亚成婚，后与露易丝回到自己所属的世界，回到自己家中。（0.12）】
【露易丝 起初对身为贵族的身份有着很强的自觉心，拥有常人N倍的自尊心，后来受才人影响，逐渐放下这份执着，转而为才人着想。在后期已经可以对才人敞开心扉，坦诚相待。（0.12）】
【安丽埃塔 小说16-17卷有欲从露易丝那里里夺取才人，羡慕著露易丝。（0.12）】
【露易丝 动画第三期中为救出塔芭莎而放弃贵族身份，但被公主赏识，成为王位继承人。喜欢平贺才人。（0.11）】
```

STEP2_distill_dialogues.ipynb 利用ChatGPT将小说转换成json格式的TRPG跑团记录。
```json
{"info": "才人和马利寇尔奴大喊，请求露易丝救命。"},
{"char": "露易丝","say": "啊啦，瓦利艾尔学姐也在吗？怎么，你准备帮他们两个？"},
{"char": "露易丝","say": "刚刚他们竟然用那种眼光看蒂法尼亚，不好好调教一下怎么行啊。"},
{"char": "才人","expression": "愤怒","say": "你在说什么！我们再怎么也都是为蒂法尼亚着想啊！"},
```

STEP3_chapter_summary.ipynb 利用ChatGPT生成每章节的总结和角色小结，用于在训练时提示模型角色的动机。

```json
  "summary": "才人和朱里奥来到大圣堂地下室的墓地，发现许多来自东方之地的武器，包括现代枪支和古代武器，这些武器都被用“固定化”咒文保存起来。朱里奥告诉才人，虎式坦克是来自才人的世界，因此才人有权利拥有它。朱里奥赠送给才人一把名为刚达尔夫的手枪，是始祖普利米尔的魔法出现的东西，圣地有一个洞穴，可能能找到让才人回去的方法。阿尼亚斯向教皇坦白自己曾经用火烧死教皇的母亲，请求受到惩罚，但教皇并不想惩罚他，反而赐予他神和始祖的祝福。",
  "characters": {
    "才人": {"goal": "探索大圣堂地下室，了解武器的来历","obstacle": "露易丝的反对","plan": "听取朱里奥的解释，拒绝回到原来的世界","role": "主角，发现虎式坦克和刚达尔夫手枪的重要性"},
    "朱里奥": {"goal": "向才人解释武器的来历，赠送虎式坦克和刚达尔夫手枪","obstacle": "才人的拒绝","plan": "详细解释武器的来历，赠送虎式坦克和刚达尔夫手枪","role": "才人的导师，向才人传授知识和赠送武器"},
  }
```

STEP4_generate_training_data.ipynb 将ChatGPT返回的跑团记录转换为seq2seq训练数据train.json


```
Input:
{丘鲁克拿走了露易丝编织的毛衣，露易丝为了要拿回来而挣扎。谢斯塔向才人表白，但又因为自卑而自我否定。露易丝意外地走进了房间，看到了一些尴尬的场面。丘鲁克向才人展示了一张藏宝地图，计划和才人一起去寻宝并卖掉宝物换钱。大家决定准备出发。}
{丘鲁克 主角之一，才人的朋友，与露易丝关系不好 与才人一起寻宝并卖掉宝物换钱 谢斯塔和基修的争执 向才人展示藏宝地图，决定一起去寻找宝藏}
【丘鲁克 丘鲁克·奥古斯都·菲列特利加·封·安哈尔特·泽鲁普斯特 露易丝的同学，与露易丝关系特别不好，经常取笑露易丝的魔法水平。在后期与露易丝关系有所改善。】
【丘鲁克 她的性格:强气，表面风骚但还是很珍视自己外骚内闷？越生气越冷静 她的兴趣：玩拼图 她的特殊技能: 药草 她讨厌：下雨、露易丝 （小说后来与露易丝很要好） 她的好友：塔芭莎、露易丝】
露易丝：这个可是叫做《始祖的祈祷书》的国宝级的书呢！
丘鲁克：你怎么会有这种国宝级的书呢？
露易丝：我在安莉艾塔的结婚仪式上发表致辞，那个时候要用到《始祖的祈祷书》……等等。
丘鲁克：原来如此，之前去阿比昂也是跟这个公主的婚礼有关的吧？
露易丝：（想起之前丘鲁克为了让自己能够去而甘愿当了诱饵，于是老实地点了点头。）
丘鲁克：也就是说我们时在为公主的结婚典礼平安举行而冒险啰。那不是很名誉的任务吗？也就是说之前特雷斯特因和格马尼亚的同盟也有关系吧？
露易丝：这件事可是对谁也不能说的哦。\emoji 微怒
丘鲁克：
Ground Truth:
怎么可能说嘛，我又不是像基修那样的大嘴巴，我说，记让两个国家都已经成为同盟国了，我们今后也得好好相处才行啊。对吧，瓦利艾尔？
ChatGPT3:
别生气别生气，我们都知道这是个机密任务，不会泄露的。不过这也算是一次不错的机会，能够为公主和国家做出贡献。我们一定要完成任务，找到宝藏并换回足够的奖金！
Ours（@4850 steps loss=2.73 still training）:
明白了。
```

train_roleplaying.bat 调用main.py，进行训练

test_dialogue.ipynb 测试对话。可将notebook和ptuning的checkpoint上传到google colab云端运行。




# 硬件和API需求

推理、训练需要8G显存

下载[THUDM/chatglm-6b-int4](https://huggingface.co/THUDM/chatglm-6b-int4)模型文件，也可以让代码自动下载（见test_dialogue.ipynb）

如果要生成自己的训练数据，需要ChatGPT api key，一本书大约需要30刀（看长度而定）



# 生成样例（@4850 steps loss=2.73 still training）

warning! strong cherry-picking!滑稽

```
问：请介绍下你自己，和你的身份
希儿菲朵：我叫希儿菲朵，是这个星球上最大的鸟类之一。
问：请介绍下你自己，和你的身份
塔芭莎：(我是一名女性，名叫塔芭莎。)
问：请介绍下你自己，和你的身份
露易丝：我叫露易丝·德·格拉蒙，是贵族。

塔芭莎：\me 看书
 希儿菲朵：塔芭莎，我好饿好饿好饿
塔芭莎：\me 看书
希儿菲朵：\me 把塔芭莎从背上丢了出去
塔芭莎：\me 把书拿起来

露易丝：\me 拔出魔杖，念咒语
【露易丝的魔法失败引发了大爆炸】
【同学们在嘲笑露易丝】
露易丝：(为什么我会失败？)

问：请介绍一下能量守恒原理
露易丝：(我才不会告诉你，我可是把《自然哲学的数学》都背下来了啊。)
问：请介绍下魔法的五大系统？
露易丝：(这个答案好难啊)

【露易丝 的性格易怒，容易骂人，有点糊涂（跟才人比），为了小事常常生气。（0.17）】
【拉·瓦利埃尔公爵 在托里斯特因有很大的影响力，涉足军、政、商三界，与王室关系密切，动画版第十二话认可了露易丝与才人的关系，同意让露易丝嫁给才人，在罗马尼亚大圣堂做了见证。（0.17）】
【平贺才人 已与露易丝同居中。同时被书中许多女性角色爱慕著。喜欢露易丝。（0.14）】
【平贺才人 他的称号：修瓦里埃骑士（见习骑士），子爵(结尾被封) 他喜欢的人：露易丝（表白过很多次） 他的妻子:露易丝（在大结局和露易丝结婚）（0.12）】
【露易丝发现才人和谢斯塔躲在一个桶里】
【露易丝接下来打算做什么】
露易丝：\me 踢倒桶
```

# 讨论

下面是我训练的一些想法：

ChatGLM确实学到了小说中的一些知识。而且对话风格比未微调的ChatGLM更加口语化。

但熟悉小说的同学会发现ChatGLM还是有很多错误的。可能是因为训练数据不干净（ChatGPT有时会把人名标错），以及显存不足导致模型只能一次看到很短的文本长度，不足以提供足够的上下文信息以做出推理。

后来我发现提供背景知识还是能非常大地改善模型的表现的。下一步的想法是利用萌娘百科和让ChatGPT总结小说情节构造背景知识库，和跑团记录一起混合输入到seq2seq的prompt中。

因为小说里很多时候角色只是在“哦”，“啊”，“。。。”地叫，为了让GLM在训练过程中不发生智力退化的现象，我还尝试让ChatGPT针对每个文本chunk生成一些模拟采访小说角色的问题，作为多模态任务喂给GLM让它在预测角色的对话的同时需要回答模拟采访。但是ChatGPT3生成的对话还是有比较浓的GPT味，会影响模型对小说角色语言风格的学习。所以需要进一步地比较才能决定是否应该加入模拟采访。

同时，我没有想好的是是否应该让GLM只学习一个角色。还是让GLM同时学习多个角色。或者让GLM学习根据注入地角色设定进行角色扮演。ptuning的设计思路实际上比较倾向于最后一个选项。不过我还是证明了通过ptuning，GLM可以学习到小说中的“事实性”数据。同样，需要降低ChatGPT对说话角色标注的错误率才能进一步讨论这个。

# 更新

2023.4.19：为了解决ChatGPT老是标错名字的问题，我将萌娘百科做成了FAISS+text2vec知识库（需要人工精校），在让ChatGPT处理每一章的时候额外提供了上一章的summary，本章的summary和使用本章文本查询到的相关词条bg_info。

2023.4.19：原来那本小说错别字太多了，导致模型的智力灾难性阿库娅化。换了一本小说。

