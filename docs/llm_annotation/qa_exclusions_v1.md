# 排除项质检说明（给 GPT，qa_v1）

你是质检员。附件有四个文件：《中文网络小说名字识别标注规则》（guide.md，下称“规则”）、章节原文（chapters.md）、要核对的地方（places.md）和这份说明。

现在的答案在 places.md 列出的每一处都**没有标名字**：裁决写“不标”的，答案认为这里不是名字；写“不计分”的，当时按规则和材料判不了，这一段对所有系统都不计分（规则 E2）。请逐处判断：只凭规则和给你的原文，这个排除站不站得住。只输出 JSON。

## 只能依据什么

- 规则、chapters.md 里的章节原文，以及每处附的“读者的原话”。
- “读者的原话”是读过全书的人对这个名字说的话，原话照抄，当作有出处的跨章证据（规则 E1），用来确认对象是什么、在书里怎么被称呼。原话里对标不标、标哪一类的倾向，要按规则核对，不能直接当结论。没有原话的地方只看规则和原文。
- 不要凭记忆补充这本小说的情节、人物、道具或设定，也不要上网查。你可能读过这本书，但这次只看给你的文字。
- 不要猜这些裁决是谁做的、怎么做出来的，也不要猜任何模型在这里标了什么、得了多少分：这些都不在你的材料里，也和你的判断无关。你只回答“按规则，这里该不该标”。

## 怎么核对

每一处单独判断，同一串字在不同位置可以有不同结果（规则 C1）。先问这一处是不是在使用名字（规则 A 部分），再看该框多长（B 部分）、是哪一类（C 部分）。

每处给一个 `verdict`：

- `"agree"`：按规则，这里**不应该标**名字（泛称、代词、普通物件、职业身份、描述、等级加类别等），排除是对的。
- `"disagree"`：按规则和原文，这里**应该标**，并在 `would_mark_as` 写类型。若该框的字串比列出的长或短，`would_mark_as` 仍写类型，在 `reason` 里写出该框的原文。
- `"cannot_tell"`：只靠给你的文字确定不了（需要别章证据，或规则确实没覆盖）。在 `reason` 里写清缺什么。

裁决是“不计分”的地方，照样按上面三个答案回答：你认为明显不是名字就 `agree`，明显是名字就 `disagree`，确实判不了就 `cannot_tell`。

每处还要写：

- `rule`：依据的规则条款，如 `"A1"`、`"A7"`、`"B1"`、`"C2"`、`"E4"`；多条用逗号隔开；没有合适条款时写 `"无"`。
- `would_mark_as`：`verdict` 是 `"disagree"` 时写六类之一（person、organization、location、dungeon、item、trait），否则写 `null`。
- `reason`：一句话理由。

## 怎么定位

places.md 每处给了章节编号（`chapter_id`，如 `ch0057`，原文里每章标题行写的编号）、段落编号（`paragraph_id`，如 `p0003`，原文里每段开头方括号里的编号）、原文字串（`text`）和 `occurrence`：这串字在这一段里从左往右第几次出现，从 1 开始数；同一段里每一次出现都数，有重叠出现时每个起点各算一次。输出时把这四项**原样抄回**，不要改字、改编号。

## 输出格式

只输出一个 JSON 对象，前后不要加任何文字；places.md 里列了几处就写几条，不多不少，不要新增别的地方：

```json
{
  "instructions_version": "qa_v1",
  "guide_version": "1.1",
  "places": [
    {"chapter_id": "ch9999", "paragraph_id": "p0002", "text": "大学生", "occurrence": 1,
     "verdict": "agree", "rule": "A1", "would_mark_as": null, "reason": "本处是身份描述，不是名字"},
    {"chapter_id": "ch9999", "paragraph_id": "p0005", "text": "回音铃", "occurrence": 2,
     "verdict": "disagree", "rule": "A2,C2", "would_mark_as": "item", "reason": "本段写明“道具回音铃”，是命名物品"},
    {"chapter_id": "ch9999", "paragraph_id": "p0008", "text": "提箱人", "occurrence": 1,
     "verdict": "cannot_tell", "rule": "A7", "would_mark_as": null, "reason": "本章分不清是专称还是职业回指，需要别章证据"}
  ]
}
```

（例子里的名字都是虚构的。）

## 输出前自查

1. places.md 里的每一处都有一条，`chapter_id`、`paragraph_id`、`text`、`occurrence` 和列出的一样；没有多出来的地方。
2. `verdict` 只用 `agree`、`disagree`、`cannot_tell` 三个值。
3. `would_mark_as` 只在 `disagree` 时写六个英文代码之一，其余写 `null`。
4. 只输出 JSON。
