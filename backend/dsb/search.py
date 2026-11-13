"""Feeling-aware phrase search: expands everyday Japanese queries ("キックしてるモーション", "楽しそう", "可愛らしい感じ", etc.) into motion attributes."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Katakana -> Hiragana so "カワイイ", "かわいい" and "可愛い" normalize cleanly.
_KATA_TO_HIRA = str.maketrans({chr(code): chr(code - 0x60) for code in range(0x30A1, 0x30F7)})

# Conversational suffixes, generic motion nouns, and particles attached to search queries.
_META_OR_VERB_RE = re.compile(
    r"(?:モーション|もーしょん|ダンス|だんす|ムーブ|むーぶ|フレーズ|ふれーず|振り付け|振付|ふりつけ|動き|うごき|"
    r"している|してる|してた|して|した|する|される|ような|ように|のある|がある|の入った|が入った|入りの|入り|"
    r"系の|風の|的な|めの|めな|っぽい|っぽく|っぽさ|らしい|らしく|らしさ|そうな|そうに|そう|"
    r"みたいな|みたいに|みたい|な感じ|の感じ|感じ|な雰囲気|雰囲気|気味|調|やで|だよ|だな|かな|とか|など|なんて)$"
)
# Hiragana particles attached directly after Katakana or Kanji (evaluated BEFORE kata->hira so "ハート"/"ヒット" stay intact).
_KATA_KANJI_PARTICLE_RE = re.compile(
    r"(?<=[ァ-ヶー一-龠々])(?:から|まで|だけ|のみ|でも|では|には|とは|での|への|や|で|の|を|に|が|は|も|と)$"
)
# Hiragana trailing "や" / "で" on hiragana tokens (excluding ~やか adjectives and 派手/腕).
_HIRA_PARTICLE_RE = re.compile(r"(?<!しな|おだ|ゆる|つや|さわ|はな|かる|あざ|にぎ)(?:やで|とか|など|や)$")
_FILLER_RE = re.compile(r"(?:め|系|風|的|感)$")

_NOISE_WORDS = {
    "ちょっと", "少し", "すごく", "すごい", "めっちゃ", "とても", "かなり", "なんか", "もっと", "一番",
    "例えば", "たとえば", "モーション", "もーしょん", "ダンス", "だんす", "動き", "うごき",
    "振り", "ふり", "振り付け", "振付", "ふりつけ", "フレーズ", "ふれーず", "ある", "いる",
}

_CONNECTOR_SPLIT_RE = re.compile(
    r"(?:(?<=[^\s]{2})(?:やで|とか|または|および)(?=[^\s]))"
    r"|(?:(?<=[ァ-ヶー一-龠々])(?:や|と|で)(?=[ァ-ヶー一-龠々]))"
)
_OR_HINT_RE = re.compile(
    r"[|｜]|(?:やで|とか|または)|(?:(?<=[ァ-ヶー一-龠々])や(?=[ァ-ヶー一-龠々]))"
)


def normalize_token(text: str) -> str:
    return text.strip().lower().translate(_KATA_TO_HIRA)


def clean_query_token(raw: str) -> str:
    """Strip conversational suffixes, generic motion words ('モーション', 'してる'), and trailing particles safely."""
    s = raw.strip()
    for _ in range(4):
        prev = s
        nxt = _META_OR_VERB_RE.sub("", s)
        if nxt != s and len(nxt) >= 2:
            s = nxt
        nxt = _KATA_KANJI_PARTICLE_RE.sub("", s)
        if nxt != s and len(nxt) >= 2:
            s = nxt
        nxt = _HIRA_PARTICLE_RE.sub("", s)
        if nxt != s and len(nxt) >= 2:
            s = nxt
        if s == prev:
            break
    return s


def stem_token(token: str) -> str:
    cleaned = clean_query_token(token)
    s = normalize_token(cleaned)
    for _ in range(3):
        prev = s
        nxt = _META_OR_VERB_RE.sub("", s)
        if nxt != s and len(nxt) >= 2:
            s = nxt
        nxt = _FILLER_RE.sub("", s)
        if nxt != s and len(nxt) >= 2:
            s = nxt
        if s == prev:
            break
    if len(s) >= 3 and s.endswith(("い", "な", "に", "く", "さ")):
        return s[:-1]
    return s


@dataclass(frozen=True)
class FeelingRule:
    triggers: tuple[str, ...]
    categories: tuple[str, ...] = ()
    moods: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    exclude_keywords: tuple[str, ...] = ()
    min_energy: int = 1
    max_energy: int = 5
    bars: tuple[int, ...] = ()


FEELING_RULES: tuple[FeelingRule, ...] = (
    # ---- Concrete Actions & Body Parts ----
    # キック・蹴り
    FeelingRule(
        triggers=("キック", "きっく", "ハイキック", "サイドキック", "フロントキック", "蹴", "けり", "ける", "けっ"),
        keywords=("キック", "ハイキック", "サイドキック", "フロントキック", "キックアップ", "蹴"),
    ),
    # 足上げ・ニーアップ
    FeelingRule(
        triggers=("足上げ", "あしあげ", "脚上げ", "ニーアップ", "膝上げ", "ひざあげ"),
        keywords=("足上げ", "ニーアップ", "キック", "ハイキック"),
    ),
    # ジャンプ・跳ねる・ホップ・バウンス
    FeelingRule(
        triggers=("ジャンプ", "じゃんぷ", "跳", "とぶ", "とんで", "はね", "ホップ", "ほっぷ", "バウンス", "ばうんす"),
        keywords=("ジャンプ", "バウンス", "跳ね", "ホップ"),
    ),
    # スクワット・しゃがみ・ドロップ
    FeelingRule(
        triggers=("スクワット", "すくわっと", "しゃが", "沈", "しずむ", "しずんで", "プリエ", "ぷりえ", "ドロップ", "どろっぷ", "低重心"),
        keywords=("スクワット", "しゃがみ", "深く沈", "低く落と", "床近くまで沈", "プリエ"),
    ),
    # ウェーブ・ボディロール
    FeelingRule(
        triggers=("ウェーブ", "うぇーぶ", "波", "ボディロール", "ぼでぃろーる", "アームウェーブ", "ロールアップ", "s字", "うねり"),
        categories=("wave",),
        keywords=("ウェーブ", "ボディロール", "アームウェーブ", "S字", "ロールアップ"),
    ),
    # 腰・ヒップ・スウェイ
    FeelingRule(
        triggers=("腰", "こし", "ヒップ", "ひっぷ", "スウェイ", "すうぇい"),
        keywords=("腰", "ヒップ", "スウェイ"),
    ),
    # ハート
    FeelingRule(
        triggers=("ハート", "はーと", "ラブ", "らぶ", "指ハート", "頬ハート"),
        keywords=("ハート",),
    ),
    # ピース
    FeelingRule(
        triggers=("ピース", "ぴーす", "ダブルピース", "vサイン"),
        keywords=("ピース",),
    ),
    # 指差し・ポイント
    FeelingRule(
        triggers=("指差", "ゆびさ", "ポイント", "ぽいんと"),
        keywords=("指差し", "ポイント"),
    ),
    # クラップ・拍手・手拍子
    FeelingRule(
        triggers=("クラップ", "くらっぷ", "拍手", "はくしゅ", "手拍子", "てびょうし"),
        keywords=("クラップ",),
    ),
    # パンチ・ガッツ・ヒット・拳
    FeelingRule(
        triggers=("パンチ", "ぱんち", "拳", "こぶし", "ガッツ", "がっつ", "ヒット", "ひっと"),
        keywords=("パンチ", "ヒット", "ガッツ", "拳"),
    ),
    # 敬礼・キラキラ・猫の手・顔フレーム・髪・振り返り・バランス
    FeelingRule(
        triggers=("敬礼", "けいれい", "おでこ"),
        keywords=("敬礼", "おでこ"),
    ),
    FeelingRule(
        triggers=("キラキラ", "きらきら", "ひらひら"),
        keywords=("キラキラ", "ひらひら"),
    ),
    FeelingRule(
        triggers=("猫", "ねこ", "にゃん", "肉球"),
        keywords=("猫の手",),
    ),
    FeelingRule(
        triggers=("髪", "かみ", "ヘア", "へあ", "かきあげ"),
        keywords=("ヘアフリップ", "髪"),
    ),
    FeelingRule(
        triggers=("振り返", "ふりかえ", "バックルック", "背中越し", "肩越し"),
        keywords=("振り返り", "振り返る", "バックルック"),
    ),
    FeelingRule(
        triggers=("片足", "かたあし", "バランス", "ばらんす"),
        keywords=("片足", "バランス"),
    ),
    FeelingRule(
        triggers=("アイソレ", "あいそれ", "首", "肩", "胸"),
        keywords=("アイソレーション", "胸", "肩", "首"),
    ),

    # ---- Genres ----
    FeelingRule(
        triggers=("k-pop", "kpop", "ケーポップ", "けーぽっぷ", "韓国", "ガルクラ", "ガールクラッシュ"),
        keywords=("K-POP", "ワッキング", "ヴォーグ", "アイソレーション"),
    ),
    FeelingRule(
        triggers=("j-pop", "jpop", "ジェイポップ", "じぇいぽっぷ", "邦楽"),
        keywords=("J-POP", "アイドル系", "ステップタッチ"),
    ),
    FeelingRule(
        triggers=("ヒップホップ", "ひっぷほっぷ", "hiphop", "ストリート", "すとりーと", "スワッグ", "b系"),
        keywords=("ヒップホップ", "ストリート", "ファンク", "ロック", "バウンス", "ランニングマン"),
    ),
    FeelingRule(
        triggers=("ハウス", "はうす", "edm", "クラブ", "くらぶ", "エレクトロ"),
        categories=("foot",),
        keywords=("ハウス", "EDM", "シャッフル", "ヒール"),
    ),
    FeelingRule(
        triggers=("ファンク", "ふぁんく", "ロックダンス"),
        keywords=("ファンク", "ロック"),
    ),
    FeelingRule(
        triggers=("ワッキング", "わっきんぐ", "ヴォーグ", "ヴぉーぐ", "ボーグ"),
        keywords=("ワッキング", "ヴォーグ", "アーム"),
    ),
    FeelingRule(
        triggers=("バラード", "ばらーど", "スロー曲"),
        categories=("wave", "pose"),
        keywords=("バラード", "余韻", "差し伸べる"),
        max_energy=3,
    ),

    # ---- Subjective Impressions & Musical Roles ----
    # 楽しそう・明るい・元気・ポップ・ノリノリ
    FeelingRule(
        triggers=("楽し", "たのし", "明る", "あかる", "ハッピー", "はっぴー", "陽気", "ウキウキ", "うきうき", "弾む", "はずむ",
                  "ポップ", "ぽっぷ", "ノリノリ", "のりのり", "元気", "げんき", "軽快", "けいかい", "盛り上が", "もりあが", "わいわい", "パーティー"),
        categories=("groove", "foot"),
        keywords=("ステップタッチ", "クラップ", "ジャンプ", "跳ね", "バウンス", "シャッフル", "ランニングマン", "ピース",
                  "キラキラ", "ハート", "J-POP", "アイドル系", "サビ", "コンボ", "ボックスステップ", "スピン"),
        exclude_keywords=("バラード", "アウトロ", "動きを収めて"),
        min_energy=2,
    ),
    # 可愛らしい・かわいい・キュート・あざとい・アイドル
    FeelingRule(
        triggers=("可愛", "かわい", "キュート", "きゅーと", "あざと", "アイドル", "あいどる", "萌え", "もえ", "女の子", "おんなのこ",
                  "ガーリー", "がーりー", "ラブリー", "らぶりー", "ぶりっ子", "ぶりっこ", "愛嬌", "あいきょう", "甘", "あま", "ちび", "ぷり"),
        moods=("女性的",),
        keywords=("アイドル系", "ハート", "ピース", "猫の手", "キラキラ", "頬", "首をかしげ", "片足重心", "顔フレーム",
                  "投げキス", "うさぎ", "客席", "くるっと回って"),
        exclude_keywords=("男性的", "拳を握り", "重く低く"),
        max_energy=4,
    ),
    # かっこいい・クール・スタイリッシュ・キレキレ
    FeelingRule(
        triggers=("かっこい", "カッコい", "格好い", "かっこうい", "クール", "くーる", "スタイリッシュ", "すたいりっしゅ",
                  "キレ", "きれ", "鋭", "するど", "イケメン", "いけめん", "男前", "おとこまえ", "渋", "しぶ", "スマート", "ハード", "はーど"),
        moods=("男性的",),
        categories=("power", "foot"),
        keywords=("K-POP", "ヒップホップ", "ヒット", "アイソレーション", "ロック", "ワッキング", "ヴォーグ", "指差し",
                  "キメ歩き", "ターン", "直線", "鋭く", "キック"),
        exclude_keywords=("猫の手", "頬に添えて"),
        min_energy=3,
    ),
    # 激しい・力強い・パワフル・爆発・ダイナミック
    FeelingRule(
        triggers=("激し", "はげし", "力強", "ちからづよ", "パワフル", "ぱわふる", "強", "つよ", "爆発", "ばくはつ",
                  "ダイナミック", "だいなみっく", "派手", "はで", "大技", "おおわざ", "熱", "あつ", "全力", "ぜんりょく", "迫力", "はくりょく"),
        categories=("power",),
        keywords=("ヒット", "爆発", "コンボ", "ジャンプ", "大回し", "サビ", "ワイドスタンス", "ロック", "ハイキック"),
        min_energy=4,
    ),
    # きれい・美しい・しなやか・セクシー・大人っぽい・優雅
    FeelingRule(
        triggers=("きれい", "綺麗", "美し", "うつくし", "しなやか", "セクシー", "せくしー", "色っぽ", "いろっぽ", "大人", "おとな",
                  "艶", "つや", "優雅", "ゆうが", "エレガント", "えれがんと", "上品", "じょうひん", "なめらか", "滑らか", "曲線", "妖艶"),
        moods=("女性的",),
        categories=("wave",),
        keywords=("ボディロール", "ウェーブ", "ヴォーグ", "S字", "流す", "腰", "バラード", "指先まで"),
        exclude_keywords=("拳", "ランニングマン"),
    ),
    # エモい・切ない・感動・バラード・叙情的
    FeelingRule(
        triggers=("エモ", "えも", "切な", "せつな", "感動", "かんどう", "叙情", "じょじょう", "ドラマチック", "どらまちっく",
                  "感情", "かんじょう", "気持ち", "きもち", "泣け", "なけ", "しっとり", "抒情", "歌い上げ", "余韻", "よいん"),
        categories=("wave", "pose"),
        keywords=("バラード", "手振り", "差し伸べる", "胸に手を当て", "顔を上げて腕を開く", "余韻", "動きを収めて", "大きな連続ウェーブ"),
        max_energy=3,
    ),
    # 落ち着く・静か・ゆったり・優しい・ゆるい
    FeelingRule(
        triggers=("落ち着", "おちつ", "静か", "しずか", "ゆったり", "ゆっくり", "穏やか", "おだやか", "優し", "やさし",
                  "控えめ", "ひかえめ", "ゆる", "リラックス", "りらっくす", "おとなし", "大人し", "ソフト", "そふと", "スロー", "すろー"),
        categories=("pose", "wave"),
        keywords=("イントロ", "アウトロ", "バラード", "揺れ", "差し伸べる", "動きを収めて"),
        max_energy=2,
    ),
    # アニメ・アニソン・ボカロ・ライブ・ファンサ
    FeelingRule(
        triggers=("アニメ", "あにめ", "アニソン", "あにそん", "ボカロ", "ぼかろ", "ライブ", "らいぶ", "ファンサ", "ふぁんさ",
                  "レス", "客席", "きゃくせき", "ステージ", "すてーじ", "オタク", "おたく", "キャッチー", "きゃっちー", "踊ってみた"),
        keywords=("アイドル系", "J-POP", "アニソン", "キメ", "ハート", "ピース", "キラキラ", "猫の手", "指差し", "スピン",
                  "客席", "ジャンプ", "クラップ", "サビ", "ステップタッチ", "コンボ", "敬礼"),
    ),
    # 決めポーズ・キメ・フィニッシュ・止まる
    FeelingRule(
        triggers=("決め", "きめ", "キメ", "ポーズ", "ぽーず", "フィニッシュ", "ふぃにっしゅ", "締め", "しめ", "ラスト", "らすと",
                  "止ま", "とま", "ストップ", "すとっぷ", "アピール", "あぴーる", "見せ場", "みせば"),
        categories=("pose",),
        keywords=("キメ", "止まる", "ターン", "スピン", "ハート", "ピース", "指差し", "両腕を大きく開く", "コンボ", "アウトロ"),
    ),
    # イントロ / アウトロ / サビ / 溜め
    FeelingRule(
        triggers=("始まり", "はじまり", "最初", "さいしょ", "出だし", "でだし", "オープニング", "おーぷにんぐ", "導入", "イントロ", "いんとろ"),
        keywords=("イントロ", "顔を上げて", "左右に揺れて"),
        max_energy=2,
    ),
    FeelingRule(
        triggers=("終わり", "おわり", "最後", "さいご", "エンディング", "えんでぃんぐ", "終盤", "アウトロ", "あうとろ"),
        keywords=("アウトロ", "動きを収めて", "止まる"),
        max_energy=2,
    ),
    FeelingRule(
        triggers=("サビ", "さび", "盛り上がり", "クライマックス"),
        categories=("power", "groove"),
        keywords=("サビ", "コンボ", "爆発", "ヒット", "ジャンプ", "ターン"),
        min_energy=3,
    ),
    FeelingRule(
        triggers=("溜め", "ため", "ビルド", "びるど", "上昇", "じょうしょう", "Bメロ", "bめろ", "サビ前", "さびまえ"),
        categories=("wave",),
        keywords=("ビルドアップ", "溜め", "徐々に立ち上がる", "爆発", "ボディロール", "ウェーブ"),
    ),
    # 回る・ターン・スピン
    FeelingRule(
        triggers=("回", "まわ", "回転", "かいてん", "ターン", "たーん", "スピン", "すぴん", "くるっ"),
        keywords=("ターン", "スピン", "回転", "回って"),
    ),
    # 移動・歩く・ステップ
    FeelingRule(
        triggers=("移動", "いどう", "歩", "ある", "ウォーク", "うぉーく", "前進", "ぜんしん", "後退", "こうたい", "横移動", "よこいどう"),
        keywords=("移動", "前進", "後退", "横移動", "ウォーク", "歩く", "下がる"),
    ),
    # 足さばき・ステップ
    FeelingRule(
        triggers=("足技", "あしわざ", "ステップ", "すてっぷ", "フットワーク", "ふっとわーく", "シャッフル", "しゃっふる", "ランニングマン"),
        categories=("foot",),
        keywords=("ステップ", "シャッフル", "ランニングマン", "ボックスステップ", "ステップタッチ", "ヒール", "足技"),
    ),
    # 手振り・指
    FeelingRule(
        triggers=("手振り", "てぶり", "手", "指", "ゆび", "ハンド", "はんど", "ジェスチャー"),
        keywords=("手振り", "ハート", "ピース", "指差し", "キラキラ", "猫の手", "顔フレーム", "クラップ", "差し伸べる", "ワッキング", "ヴォーグ"),
    ),
)


def impression_words(item: dict) -> list[str]:
    """Natural Japanese impression & action synonyms derived from a phrase's metadata for EmbeddingGemma 2 & lexical search."""
    tags = set(item.get("tags", []))
    desc = item.get("description", "")
    cat = item.get("category", "")
    energy = int(item.get("energy", 3))
    words: list[str] = []

    if "キック" in tags or "蹴" in desc:
        words.extend(["キック", "蹴り", "蹴る", "足上げ", "キックしてるモーション"])
    if "足上げ" in tags:
        words.extend(["足上げ", "脚上げ", "ニーアップ"])
    if "ジャンプ" in tags:
        words.extend(["ジャンプ", "跳ねる", "跳ぶ", "ホップ"])
    if "スクワット" in tags:
        words.extend(["スクワット", "しゃがみ", "しゃがむ", "ドロップ"])
    if "ターン" in tags or "回転" in tags or "スピン" in tags:
        words.extend(["ターン", "回転", "スピン", "回る"])
    if "アイドル系" in tags or any(k in desc for k in ("ハート", "ピース", "猫の手", "キラキラ", "頬", "首をかしげ")):
        words.extend(["かわいい", "可愛らしい", "キュート", "アイドルらしい", "ファンサ", "アニメっぽい", "キャッチー"])
    if "女性的" in tags:
        words.extend(["しなやか", "柔らかい", "かわいらしい"])
        if cat == "wave" or "S字" in desc or "腰" in desc:
            words.extend(["きれい", "美しい", "大人っぽい", "セクシー", "優雅"])
    if "男性的" in tags:
        words.extend(["かっこいい", "クール", "力強い", "ワイルド", "スタイリッシュ"])
    if "K-POP" in tags or "ヒップホップ" in tags:
        words.extend(["かっこいい", "スタイリッシュ", "キレのある", "ガルクラ"])
    if "J-POP" in tags or (cat in ("groove", "foot") and energy >= 3 and "バラード" not in tags):
        words.extend(["楽しそう", "楽しい", "明るい", "元気", "ノリノリ", "ポップ", "軽快"])
    if "バラード" in tags or any(k in desc for k in ("差し伸べる", "胸に手を当て", "顔を上げて腕を開く", "動きを収めて")):
        words.extend(["エモい", "切ない", "叙情的", "感動的", "優しい"])
    if energy <= 2:
        words.extend(["落ち着いた", "静か", "ゆったり", "控えめ"])
    if energy >= 4 or cat == "power":
        words.extend(["激しい", "パワフル", "盛り上がる", "ダイナミック", "派手"])
    if cat == "pose" or "キメ" in tags or "ターン" in tags or "スピン" in tags:
        words.extend(["決めポーズ", "キメ", "ポーズ", "アピール"])
    seen: set[str] = set()
    return [w for w in words if not (w in seen or seen.add(w))]


def _split_query(query: str) -> tuple[list[str], bool]:
    raw = query.strip()
    if not raw:
        return [], False
    is_or = bool(_OR_HINT_RE.search(raw))
    pre_split = _CONNECTOR_SPLIT_RE.sub(" ", raw)
    cleaned = re.sub(r"[、，,。・/／|｜+＋\s　]+", " ", pre_split)
    parts = [p.strip() for p in cleaned.split() if p.strip()]
    tokens: list[str] = []
    seen: set[str] = set()
    for p in parts:
        tok = clean_query_token(p)
        norm = normalize_token(tok)
        if not norm or norm in _NOISE_WORDS:
            continue
        if norm not in seen:
            seen.add(norm)
            tokens.append(tok)
    if not tokens:
        fallback = [clean_query_token(p) for p in parts if clean_query_token(p)]
        return fallback, is_or
    return tokens, is_or


def _score_token(item: dict, raw_token: str, category_label: str) -> float:
    cleaned = clean_query_token(raw_token)
    norm = normalize_token(cleaned)
    stem = stem_token(cleaned)
    if not norm:
        return 0.0
    tags = item.get("tags", [])
    norm_tags = [normalize_token(t) for t in tags]
    haystack_raw = " ".join([
        item.get("name", ""),
        item.get("description", ""),
        category_label,
        *tags,
        *impression_words(item),
    ])
    haystack = normalize_token(haystack_raw)

    score = 0.0
    if norm in norm_tags or (len(stem) >= 2 and stem in norm_tags):
        score = 1.25
    elif norm in haystack:
        score = 1.05
    elif len(stem) >= 2 and stem in haystack:
        score = 0.90

    blob = f"{item.get('name', '')} {item.get('description', '')} {' '.join(tags)}"
    energy = int(item.get("energy", 3))
    cat = item.get("category", "")

    for rule in FEELING_RULES:
        matched_rule = any(
            normalize_token(trig) in norm or (len(stem) >= 2 and (normalize_token(trig) in stem or stem in normalize_token(trig)))
            for trig in rule.triggers
        )
        if not matched_rule:
            continue
        if any(ex in blob for ex in rule.exclude_keywords):
            continue
        pts = 0.0
        if rule.moods and any(m in tags for m in rule.moods):
            pts += 0.45
        if rule.categories and cat in rule.categories:
            pts += 0.38
        kw_hits = sum(1 for kw in rule.keywords if kw in blob)
        if kw_hits:
            pts += min(0.85, 0.45 + 0.16 * kw_hits)
        elif not rule.moods and not rule.categories:
            continue
        if rule.min_energy <= energy <= rule.max_energy:
            if rule.min_energy > 1 or rule.max_energy < 5:
                pts += 0.22
        else:
            pts *= 0.25
        if rule.bars and int(item.get("bars", 2)) in rule.bars:
            pts += 0.25
        score = max(score, pts)

    return score


def feeling_scores(query: str, items: list[dict], categories: dict[str, str]) -> dict[str, float]:
    """Returns {phrase_id: score >= 0} for a free-form / impression / action query."""
    tokens, is_or = _split_query(query)
    if not tokens:
        return {item["id"]: 0.0 for item in items}
    per_item_tokens: dict[str, list[float]] = {}
    any_full_match = False
    for item in items:
        cat_label = categories.get(item.get("category", ""), "")
        pt = [_score_token(item, tok, cat_label) for tok in tokens]
        per_item_tokens[item["id"]] = pt
        if all(s > 0.0 for s in pt):
            any_full_match = True

    result: dict[str, float] = {}
    for item in items:
        pt = per_item_tokens[item["id"]]
        if all(s > 0.0 for s in pt):
            result[item["id"]] = float(np.mean(pt))
        elif any(s > 0.0 for s in pt) and len(tokens) > 1 and (is_or or not any_full_match):
            result[item["id"]] = float(max(pt)) * 0.85 if is_or else float(np.mean(pt)) * 0.45
        else:
            result[item["id"]] = 0.0
    return result
