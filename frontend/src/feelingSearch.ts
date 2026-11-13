import { Clip, Phrase, S, clipDuration, clips, phraseOf } from "./store";

const META_OR_VERB_RE = /(?:モーション|もーしょん|ダンス|だんす|ムーブ|むーぶ|フレーズ|ふれーず|振り付け|振付|ふりつけ|動き|うごき|している|してる|してた|して|した|する|される|ような|ように|のある|がある|の入った|が入った|入りの|入り|系の|風の|的な|めの|めな|っぽい|っぽく|っぽさ|らしい|らしく|らしさ|そうな|そうに|そう|みたいな|みたいに|みたい|な感じ|の感じ|感じ|な雰囲気|雰囲気|気味|調|やで|だよ|だな|かな|とか|など|なんて)$/;
const KATA_KANJI_PARTICLE_RE = /([ァ-ヶー一-龠々])(?:から|まで|だけ|のみ|でも|では|には|とは|での|への|や|で|の|を|に|が|は|も|と)$/;
const HIRA_YAKA_END_RE = /(?:しなや|おだや|ゆるや|つやや|さわや|はなや|かるや|あざや|にぎや)$/;
const FILLER_RE = /(?:め|系|風|的|感)$/;
const NOISE_WORDS = new Set([
  "ちょっと", "少し", "すごく", "すごい", "めっちゃ", "とても", "かなり", "なんか", "もっと", "一番",
  "例えば", "たとえば", "モーション", "もーしょん", "ダンス", "だんす", "動き", "うごき",
  "振り", "ふり", "振り付け", "振付", "ふりつけ", "フレーズ", "ふれーず", "ある", "いる",
]);

function kataToHira(text: string): string {
  let out = "";
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    out += code >= 0x30a1 && code <= 0x30f6 ? String.fromCharCode(code - 0x60) : text[i];
  }
  return out;
}

export function normalizeToken(text: string): string {
  return kataToHira(text.trim().toLowerCase());
}

export function cleanQueryToken(raw: string): string {
  let s = raw.trim();
  for (let i = 0; i < 4; i++) {
    const prev = s;
    let nxt = s.replace(META_OR_VERB_RE, "");
    if (nxt !== s && nxt.length >= 2) { s = nxt; }
    nxt = s.replace(KATA_KANJI_PARTICLE_RE, "$1");
    if (nxt !== s && nxt.length >= 2) { s = nxt; }
    if (s.length >= 3 && s.endsWith("や") && !HIRA_YAKA_END_RE.test(s)) {
      s = s.slice(0, -1);
    }
    if (s === prev) { break; }
  }
  return s;
}

function stemToken(token: string): string {
  let s = normalizeToken(cleanQueryToken(token));
  for (let i = 0; i < 3; i++) {
    const prev = s;
    let nxt = s.replace(META_OR_VERB_RE, "");
    if (nxt !== s && nxt.length >= 2) { s = nxt; }
    nxt = s.replace(FILLER_RE, "");
    if (nxt !== s && nxt.length >= 2) { s = nxt; }
    if (s === prev) { break; }
  }
  if (s.length >= 3 && /[いなにくさ]$/.test(s)) {
    return s.slice(0, -1);
  }
  return s;
}

interface FeelingRule {
  triggers: string[];
  categories?: string[];
  moods?: string[];
  keywords?: string[];
  exclude?: string[];
  minEnergy?: number;
  maxEnergy?: number;
}

const FEELING_RULES: FeelingRule[] = [
  // Concrete actions & body parts
  {
    triggers: ["キック", "きっく", "ハイキック", "サイドキック", "フロントキック", "蹴", "けり", "ける", "けっ"],
    keywords: ["キック", "ハイキック", "サイドキック", "フロントキック", "キックアップ", "蹴"],
  },
  {
    triggers: ["足上げ", "あしあげ", "脚上げ", "ニーアップ", "膝上げ", "ひざあげ"],
    keywords: ["足上げ", "ニーアップ", "キック", "ハイキック"],
  },
  {
    triggers: ["ジャンプ", "じゃんぷ", "跳", "とぶ", "とんで", "はね", "ホップ", "ほっぷ", "バウンス", "ばうんす"],
    keywords: ["ジャンプ", "バウンス", "跳ね", "ホップ"],
  },
  {
    triggers: ["スクワット", "すくわっと", "しゃが", "沈", "しずむ", "しずんで", "プリエ", "ぷりえ", "ドロップ", "どろっぷ", "低重心"],
    keywords: ["スクワット", "しゃがみ", "深く沈", "低く落と", "床近くまで沈", "プリエ"],
  },
  {
    triggers: ["ウェーブ", "うぇーぶ", "波", "ボディロール", "ぼでぃろーる", "アームウェーブ", "ロールアップ", "s字", "うねり"],
    categories: ["wave"],
    keywords: ["ウェーブ", "ボディロール", "アームウェーブ", "S字", "ロールアップ"],
  },
  {
    triggers: ["腰", "こし", "ヒップ", "ひっぷ", "スウェイ", "すうぇい"],
    keywords: ["腰", "ヒップ", "スウェイ"],
  },
  {
    triggers: ["ハート", "はーと", "ラブ", "らぶ", "指ハート", "頬ハート"],
    keywords: ["ハート"],
  },
  {
    triggers: ["ピース", "ぴーす", "ダブルピース", "vサイン"],
    keywords: ["ピース"],
  },
  {
    triggers: ["指差", "ゆびさ", "ポイント", "ぽいんと"],
    keywords: ["指差し", "ポイント"],
  },
  {
    triggers: ["クラップ", "くらっぷ", "拍手", "はくしゅ", "手拍子", "てびょうし"],
    keywords: ["クラップ"],
  },
  {
    triggers: ["パンチ", "ぱんち", "拳", "こぶし", "ガッツ", "がっつ", "ヒット", "ひっと"],
    keywords: ["パンチ", "ヒット", "ガッツ", "拳"],
  },
  {
    triggers: ["敬礼", "けいれい", "おでこ"],
    keywords: ["敬礼", "おでこ"],
  },
  {
    triggers: ["キラキラ", "きらきら", "ひらひら"],
    keywords: ["キラキラ", "ひらひら"],
  },
  {
    triggers: ["猫", "ねこ", "にゃん", "肉球"],
    keywords: ["猫の手"],
  },
  {
    triggers: ["髪", "かみ", "ヘア", "へあ", "かきあげ"],
    keywords: ["ヘアフリップ", "髪"],
  },
  {
    triggers: ["振り返", "ふりかえ", "バックルック", "背中越し", "肩越し"],
    keywords: ["振り返り", "振り返る", "バックルック"],
  },
  {
    triggers: ["片足", "かたあし", "バランス", "ばらんす"],
    keywords: ["片足", "バランス"],
  },
  {
    triggers: ["アイソレ", "あいそれ", "首", "肩", "胸"],
    keywords: ["アイソレーション", "胸", "肩", "首"],
  },
  // Genres
  {
    triggers: ["k-pop", "kpop", "ケーポップ", "けーぽっぷ", "韓国", "ガルクラ", "ガールクラッシュ"],
    keywords: ["K-POP", "ワッキング", "ヴォーグ", "アイソレーション"],
  },
  {
    triggers: ["j-pop", "jpop", "ジェイポップ", "じぇいぽっぷ", "邦楽"],
    keywords: ["J-POP", "アイドル系", "ステップタッチ"],
  },
  {
    triggers: ["ヒップホップ", "ひっぷほっぷ", "hiphop", "ストリート", "すとりーと", "スワッグ", "b系"],
    keywords: ["ヒップホップ", "ストリート", "ファンク", "ロック", "バウンス", "ランニングマン"],
  },
  {
    triggers: ["ハウス", "はうす", "edm", "クラブ", "くらぶ", "エレクトロ"],
    categories: ["foot"],
    keywords: ["ハウス", "EDM", "シャッフル", "ヒール"],
  },
  {
    triggers: ["ファンク", "ふぁんく", "ロックダンス"],
    keywords: ["ファンク", "ロック"],
  },
  {
    triggers: ["ワッキング", "わっきんぐ", "ヴォーグ", "ヴぉーぐ", "ボーグ"],
    keywords: ["ワッキング", "ヴォーグ", "アーム"],
  },
  {
    triggers: ["バラード", "ばらーど", "スロー曲"],
    categories: ["wave", "pose"],
    keywords: ["バラード", "余韻", "差し伸べる"],
    maxEnergy: 3,
  },
  // Subjective impressions & musical roles
  {
    triggers: ["楽し", "たのし", "明る", "あかる", "ハッピー", "陽気", "ウキウキ", "弾む", "はずむ", "ポップ", "ノリノリ", "元気", "げんき", "軽快", "盛り上が", "もりあが", "わいわい", "パーティー"],
    categories: ["groove", "foot"],
    keywords: ["ステップタッチ", "クラップ", "ジャンプ", "跳ね", "バウンス", "シャッフル", "ランニングマン", "ピース", "キラキラ", "ハート", "J-POP", "アイドル系", "サビ", "コンボ", "ボックスステップ", "スピン"],
    exclude: ["バラード", "アウトロ", "動きを収めて"],
    minEnergy: 2,
  },
  {
    triggers: ["可愛", "かわい", "キュート", "あざと", "アイドル", "萌え", "もえ", "女の子", "おんなのこ", "ガーリー", "ラブリー", "ぶりっ子", "愛嬌", "あいきょう", "甘", "あま", "ちび", "ぷり"],
    moods: ["女性的"],
    keywords: ["アイドル系", "ハート", "ピース", "猫の手", "キラキラ", "頬", "首をかしげ", "片足重心", "顔フレーム", "投げキス", "うさぎ", "客席", "くるっと回って"],
    exclude: ["男性的", "拳を握り"],
    maxEnergy: 4,
  },
  {
    triggers: ["かっこい", "格好い", "かっこうい", "クール", "スタイリッシュ", "キレ", "きれ", "鋭", "するど", "イケメン", "男前", "渋", "しぶ", "スマート", "ハード"],
    moods: ["男性的"],
    categories: ["power", "foot"],
    keywords: ["K-POP", "ヒップホップ", "ヒット", "アイソレーション", "ロック", "ワッキング", "ヴォーグ", "指差し", "キメ歩き", "ターン", "直線", "鋭く", "キック"],
    exclude: ["猫の手", "頬に添えて"],
    minEnergy: 3,
  },
  {
    triggers: ["激し", "はげし", "力強", "ちからづよ", "パワフル", "強", "つよ", "爆発", "ばくはつ", "ダイナミック", "派手", "はで", "大技", "おおわざ", "熱", "あつ", "全力", "迫力"],
    categories: ["power"],
    keywords: ["ヒット", "爆発", "コンボ", "ジャンプ", "大回し", "サビ", "ワイドスタンス", "ロック", "ハイキック"],
    minEnergy: 4,
  },
  {
    triggers: ["きれい", "綺麗", "美し", "うつくし", "しなやか", "セクシー", "色っぽ", "いろっぽ", "大人", "おとな", "艶", "つや", "優雅", "ゆうが", "エレガント", "上品", "なめらか", "滑らか", "曲線", "妖艶"],
    moods: ["女性的"],
    categories: ["wave"],
    keywords: ["ボディロール", "ウェーブ", "ヴォーグ", "S字", "流す", "腰", "バラード", "指先まで"],
    exclude: ["拳", "ランニングマン"],
  },
  {
    triggers: ["エモ", "えも", "切な", "せつな", "感動", "かんどう", "叙情", "ドラマチック", "感情", "かんじょう", "気持ち", "きもち", "泣け", "しっとり", "歌い上げ", "余韻", "よいん"],
    categories: ["wave", "pose"],
    keywords: ["バラード", "手振り", "差し伸べる", "胸に手を当て", "顔を上げて腕を開く", "余韻", "動きを収めて", "大きな連続ウェーブ"],
    maxEnergy: 3,
  },
  {
    triggers: ["落ち着", "おちつ", "静か", "しずか", "ゆったり", "ゆっくり", "穏やか", "おだやか", "優し", "やさし", "控えめ", "ひかえめ", "ゆる", "リラックス", "おとなし", "大人し", "ソフト", "スロー"],
    categories: ["pose", "wave"],
    keywords: ["イントロ", "アウトロ", "バラード", "揺れ", "差し伸べる", "動きを収めて"],
    maxEnergy: 2,
  },
  {
    triggers: ["アニメ", "アニソン", "ボカロ", "ライブ", "ファンサ", "レス", "客席", "きゃくせき", "ステージ", "オタク", "キャッチー", "踊ってみた"],
    keywords: ["アイドル系", "J-POP", "アニソン", "キメ", "ハート", "ピース", "キラキラ", "猫の手", "指差し", "スピン", "客席", "ジャンプ", "クラップ", "サビ", "ステップタッチ", "コンボ", "敬礼"],
  },
  {
    triggers: ["決め", "きめ", "キメ", "ポーズ", "フィニッシュ", "締め", "しめ", "ラスト", "止ま", "とま", "ストップ", "アピール", "見せ場", "みせば"],
    categories: ["pose"],
    keywords: ["キメ", "止まる", "ターン", "スピン", "ハート", "ピース", "指差し", "両腕を大きく開く", "コンボ", "アウトロ"],
  },
  {
    triggers: ["始まり", "はじまり", "最初", "さいしょ", "出だし", "でだし", "オープニング", "導入", "イントロ"],
    keywords: ["イントロ", "顔を上げて", "左右に揺れて"],
    maxEnergy: 2,
  },
  {
    triggers: ["終わり", "おわり", "最後", "さいご", "エンディング", "終盤", "アウトロ"],
    keywords: ["アウトロ", "動きを収めて", "止まる"],
    maxEnergy: 2,
  },
  {
    triggers: ["サビ", "さび", "盛り上がり", "クライマックス"],
    categories: ["power", "groove"],
    keywords: ["サビ", "コンボ", "爆発", "ヒット", "ジャンプ", "ターン"],
    minEnergy: 3,
  },
  {
    triggers: ["溜め", "ため", "ビルド", "上昇", "じょうしょう", "bメロ", "サビ前", "さびまえ"],
    categories: ["wave"],
    keywords: ["ビルドアップ", "溜め", "徐々に立ち上がる", "爆発", "ボディロール", "ウェーブ"],
  },
  {
    triggers: ["回", "まわ", "回転", "かいてん", "ターン", "スピン", "くるっ"],
    keywords: ["ターン", "スピン", "回転", "回って"],
  },
  {
    triggers: ["移動", "いどう", "歩", "ある", "ウォーク", "前進", "後退", "横移動"],
    keywords: ["移動", "前進", "後退", "横移動", "ウォーク", "歩く", "下がる"],
  },
  {
    triggers: ["足技", "あしわざ", "ステップ", "フットワーク", "シャッフル", "ランニングマン"],
    categories: ["foot"],
    keywords: ["ステップ", "シャッフル", "ランニングマン", "ボックスステップ", "ステップタッチ", "ヒール", "足技"],
  },
  {
    triggers: ["手振り", "てぶり", "手", "指", "ゆび", "ハンド", "ジェスチャー"],
    keywords: ["手振り", "ハート", "ピース", "指差し", "キラキラ", "猫の手", "顔フレーム", "クラップ", "差し伸べる", "ワッキング", "ヴォーグ"],
  },
];

function impressionWords(item: Phrase): string[] {
  const tags = new Set(item.tags ?? []);
  const desc = item.description ?? "";
  const cat = item.category ?? "";
  const energy = item.energy ?? 3;
  const out: string[] = [];
  if (tags.has("キック") || desc.includes("蹴")) {
    out.push("キック", "蹴り", "蹴る", "足上げ");
  }
  if (tags.has("足上げ")) { out.push("足上げ", "脚上げ", "ニーアップ"); }
  if (tags.has("ジャンプ")) { out.push("ジャンプ", "跳ねる", "跳ぶ", "ホップ"); }
  if (tags.has("スクワット")) { out.push("スクワット", "しゃがみ", "しゃがむ", "ドロップ"); }
  if (tags.has("ターン") || tags.has("回転") || tags.has("スピン")) { out.push("ターン", "回転", "スピン", "回る"); }
  if (tags.has("アイドル系") || ["ハート", "ピース", "猫の手", "キラキラ", "頬", "首をかしげ"].some((k) => desc.includes(k))) {
    out.push("かわいい", "可愛らしい", "キュート", "アイドルらしい", "ファンサ", "アニメっぽい", "キャッチー");
  }
  if (tags.has("女性的")) {
    out.push("しなやか", "柔らかい", "かわいらしい");
    if (cat === "wave" || desc.includes("S字") || desc.includes("腰")) {
      out.push("きれい", "美しい", "大人っぽい", "セクシー", "優雅");
    }
  }
  if (tags.has("男性的")) {
    out.push("かっこいい", "クール", "力強い", "ワイルド", "スタイリッシュ");
  }
  if (tags.has("K-POP") || tags.has("ヒップホップ")) {
    out.push("かっこいい", "スタイリッシュ", "キレのある", "ガルクラ");
  }
  if (tags.has("J-POP") || ((cat === "groove" || cat === "foot") && energy >= 3 && !tags.has("バラード"))) {
    out.push("楽しそう", "楽しい", "明るい", "元気", "ノリノリ", "ポップ", "軽快");
  }
  if (tags.has("バラード") || ["差し伸べる", "胸に手を当て", "顔を上げて腕を開く", "動きを収めて"].some((k) => desc.includes(k))) {
    out.push("エモい", "切ない", "叙情的", "感動的", "優しい");
  }
  if (energy <= 2) { out.push("落ち着いた", "静か", "ゆったり", "控えめ"); }
  if (energy >= 4 || cat === "power") { out.push("激しい", "パワフル", "盛り上がる", "ダイナミック", "派手"); }
  if (cat === "pose" || tags.has("キメ") || tags.has("ターン") || tags.has("スピン")) {
    out.push("決めポーズ", "キメ", "ポーズ", "アピール");
  }
  return out;
}

function splitQuery(query: string): { tokens: string[]; isOr: boolean } {
  const raw = query.trim();
  if (!raw) { return { tokens: [], isOr: false }; }
  const isOr = /[|｜]/.test(raw) || /(?:やで|とか|または)/.test(raw) || /[ァ-ヶー一-龠々]や[ァ-ヶー一-龠々]/.test(raw);
  const preSplit = raw
    .replace(/([^\s]{2})(?:やで|とか|または|および)(?=[^\s])/g, "$1 ")
    .replace(/([ァ-ヶー一-龠々])(?:や|と|で)(?=[ァ-ヶー一-龠々])/g, "$1 ");
  const cleaned = preSplit.replace(/[、，,。・/／|｜+＋\s　]+/g, " ");
  const parts = cleaned.split(" ").map((s) => s.trim()).filter(Boolean);
  const tokens: string[] = [];
  const seen = new Set<string>();
  for (const p of parts) {
    const tok = cleanQueryToken(p);
    const norm = normalizeToken(tok);
    if (!norm || NOISE_WORDS.has(norm)) { continue; }
    if (!seen.has(norm)) {
      seen.add(norm);
      tokens.push(tok);
    }
  }
  if (!tokens.length) {
    return { tokens: parts.map((p) => cleanQueryToken(p)).filter(Boolean), isOr };
  }
  return { tokens, isOr };
}

function scoreItemTokens(item: Phrase, tokens: string[], categories: Record<string, string>): number[] {
  const catLabel = categories[item.category] ?? "";
  const rawTags = item.tags ?? [];
  const normTags = new Set(rawTags.map((t) => normalizeToken(t)));
  const haystack = normalizeToken([item.name, item.description, catLabel, ...rawTags, ...impressionWords(item)].join(" "));
  const blob = `${item.name} ${item.description} ${rawTags.join(" ")}`;
  const tags = new Set(rawTags);

  return tokens.map((raw) => {
    const cleaned = cleanQueryToken(raw);
    const norm = normalizeToken(cleaned);
    const stem = stemToken(cleaned);
    if (!norm) { return 0; }
    let score = 0;
    if (normTags.has(norm) || (stem.length >= 2 && normTags.has(stem))) { score = 1.25; }
    else if (haystack.includes(norm)) { score = 1.05; }
    else if (stem.length >= 2 && haystack.includes(stem)) { score = 0.90; }

    for (const rule of FEELING_RULES) {
      const matched = rule.triggers.some((trig) => {
        const nt = normalizeToken(trig);
        return norm.includes(nt) || (stem.length >= 2 && (stem.includes(nt) || nt.includes(stem)));
      });
      if (!matched) { continue; }
      if (rule.exclude?.some((ex) => blob.includes(ex))) { continue; }
      let pts = 0;
      if (rule.moods?.some((m) => tags.has(m))) { pts += 0.45; }
      if (rule.categories?.includes(item.category)) { pts += 0.38; }
      const kwHits = (rule.keywords ?? []).filter((kw) => blob.includes(kw)).length;
      if (kwHits > 0) {
        pts += Math.min(0.85, 0.45 + 0.16 * kwHits);
      } else if (!rule.moods?.length && !rule.categories?.length) {
        continue;
      }
      const minE = rule.minEnergy ?? 1;
      const maxE = rule.maxEnergy ?? 5;
      if (item.energy >= minE && item.energy <= maxE) {
        if (minE > 1 || maxE < 5) { pts += 0.22; }
      } else {
        pts *= 0.25;
      }
      if (pts > score) { score = pts; }
    }
    return score;
  });
}

export function scoreFeelingItem(item: Phrase, query: string, categories: Record<string, string>): number {
  const { tokens, isOr } = splitQuery(query);
  if (!tokens.length) { return 1; }
  const perToken = scoreItemTokens(item, tokens, categories);
  if (perToken.every((s) => s > 0)) {
    return perToken.reduce((a, b) => a + b, 0) / perToken.length;
  }
  if (tokens.length > 1 && perToken.some((s) => s > 0)) {
    return isOr ? Math.max(...perToken) * 0.85 : (perToken.reduce((a, b) => a + b, 0) / perToken.length) * 0.45;
  }
  return 0;
}

export function filterByFeeling(items: Phrase[], query: string, categories: Record<string, string>): Phrase[] {
  const { tokens, isOr } = splitQuery(query);
  if (!tokens.length) { return items; }
  const evaluated = items.map((item) => {
    const pt = scoreItemTokens(item, tokens, categories);
    const full = pt.every((s) => s > 0);
    const any = pt.some((s) => s > 0);
    return { item, pt, full, any };
  });
  const anyFull = evaluated.some((e) => e.full);
  const scored = evaluated
    .map((e) => {
      let score = 0;
      if (e.full) {
        score = e.pt.reduce((a, b) => a + b, 0) / e.pt.length;
      } else if (e.any && tokens.length > 1 && (isOr || !anyFull)) {
        score = isOr ? Math.max(...e.pt) * 0.85 : (e.pt.reduce((a, b) => a + b, 0) / e.pt.length) * 0.45;
      }
      return { item: e.item, score };
    })
    .filter((entry) => entry.score > 0.05)
    .sort((a, b) => b.score - a.score);
  return scored.map((entry) => entry.item);
}

export const GENRE_TAG_OPTIONS = [
  "K-POP", "アイドル系", "J-POP", "アニソン", "ヒップホップ", "ストリート",
  "ハウス", "EDM", "ファンク", "バラード", "ワッキング", "ヴォーグ",
] as const;

export const MOOD_TAG_OPTIONS = ["女性的", "男性的", "中性的"] as const;

export const ACTION_TAG_OPTIONS = [
  "キック", "ハイキック", "足上げ", "ターン", "ジャンプ", "バウンス", "ステップ", "ウォーク",
  "ウェーブ", "ボディロール", "スクワット", "腰", "ハート", "ピース", "指差し", "クラップ",
  "パンチ", "ヒット", "アイソレーション", "手振り", "キラキラ", "敬礼", "バランス",
  "キメ", "サビ", "ビルドアップ", "イントロ", "アウトロ",
] as const;

const NORM_GENRE_SET = new Set(GENRE_TAG_OPTIONS.map((g) => normalizeToken(g)));
const NORM_MOOD_SET = new Set(MOOD_TAG_OPTIONS.map((m) => normalizeToken(m)));

export function activeQueryNormTokens(query: string): Set<string> {
  return new Set(splitQuery(query).tokens.map((t) => normalizeToken(t)));
}

export function cleanFilterTags(filterTags: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const raw of filterTags) {
    for (const part of String(raw).split(/[,、，・/／|｜]+/)) {
      const trimmed = part.trim();
      if (!trimmed) { continue; }
      const cleaned = cleanQueryToken(trimmed) || trimmed;
      const norm = normalizeToken(cleaned);
      if (norm && !seen.has(norm)) {
        seen.add(norm);
        out.push(cleaned);
      }
    }
  }
  return out;
}

function matchesSingleFilterTag(item: Phrase, tag: string, categories: Record<string, string>): boolean {
  const cleaned = cleanQueryToken(tag) || tag.trim();
  const norm = normalizeToken(cleaned);
  if (!norm) { return false; }
  const normTags = new Set((item.tags ?? []).map((t) => normalizeToken(t)));
  const catLabel = categories[item.category] ?? "";
  if (normTags.has(norm) || norm === normalizeToken(item.category) || (catLabel && norm === normalizeToken(catLabel))) {
    return true;
  }
  if (NORM_GENRE_SET.has(norm) || NORM_MOOD_SET.has(norm)) {
    return false;
  }
  const hay = normalizeToken(`${item.name} ${item.description}`);
  if (hay.includes(norm)) { return true; }
  return scoreItemTokens(item, [cleaned], categories)[0] >= 0.60;
}

export function filterPhrasesByTags(items: Phrase[], filterTags: string[], categories: Record<string, string>): Phrase[] {
  const cleaned = cleanFilterTags(filterTags);
  if (!cleaned.length) { return items; }
  const genres = cleaned.filter((t) => NORM_GENRE_SET.has(normalizeToken(t)));
  const moods = cleaned.filter((t) => NORM_MOOD_SET.has(normalizeToken(t)));
  const others = cleaned.filter((t) => !NORM_GENRE_SET.has(normalizeToken(t)) && !NORM_MOOD_SET.has(normalizeToken(t)));
  return items.filter((item) => {
    if (genres.length && !genres.some((g) => matchesSingleFilterTag(item, g, categories))) { return false; }
    if (moods.length && !moods.some((m) => matchesSingleFilterTag(item, m, categories))) { return false; }
    if (others.length && !others.some((o) => matchesSingleFilterTag(item, o, categories))) { return false; }
    return true;
  });
}

// ---- Musical match ranking at a timeline position (for right-click insert / replace menu)

export interface SpotMatch {
  score: number;
  percent: number;
  reason: string;
}

const KIND_LABELS: Record<string, string> = {
  intro: "イントロ",
  verse: "Aメロ",
  build: "Bメロ",
  chorus: "サビ",
  interlude: "間奏",
  outro: "アウトロ",
};

function baseStem(name: string): string {
  return name.trim().toLowerCase().replace(/[\s_-]+[mfn]$/i, "");
}

function hasEndingPose(item: Phrase): boolean {
  if (item.category === "pose") { return true; }
  const tags = new Set(item.tags ?? []);
  if (tags.has("キメ") || tags.has("アウトロ") || tags.has("スピン") || tags.has("ターン")) { return true; }
  const desc = item.description ?? "";
  return ["止まる", "動きを収めて", "フィニッシュ"].some((k) => desc.includes(k));
}

export function rankPhrasesForSpot(
  items: Phrase[],
  at: number,
  opts?: { replaceClip?: Clip }
): Map<string, SpotMatch> {
  const result = new Map<string, SpotMatch>();
  const song = S.song;
  if (!song || !items.length) { return result; }

  const barSec = song.barSeconds || 2.0;
  const section = song.sections.find((s) => at >= s.start - 1e-3 && at < s.end) ?? song.sections.at(-1);
  const kind = section?.kind ?? "verse";
  const flow = S.project?.flow?.[kind] ?? { categories: ["groove", "foot", "wave"], energy: 3, move: 0.4, text: "" };

  const barIdx = Math.max(0, Math.min((song.energyPerBar?.length ?? 1) - 1, Math.floor(at / Math.max(barSec, 0.2))));
  const localE = song.energyPerBar?.[barIdx];
  const targetEnergy = localE !== undefined ? 0.55 * flow.energy + 0.45 * (1 + 4 * Math.max(0, Math.min(1, localE))) : flow.energy;

  const allClips = clips();
  const replaceClip = opts?.replaceClip;
  const refTime = replaceClip ? replaceClip.start : at;
  const earlier = allClips.filter((c) => c.start < refTime - 1e-3 && c.id !== replaceClip?.id).slice(-6);
  const later = allClips.find((c) => c.start > refTime + 1e-3 && c.id !== replaceClip?.id);
  const recentIds = new Set(earlier.map((c) => c.phraseId));
  const lastId = earlier.at(-1)?.phraseId;
  const recentStems = earlier.map((c) => baseStem(phraseOf(c.phraseId)?.name ?? ""));

  // Dominant mood in current timeline
  const moodCounts: Record<string, number> = { 女性的: 0, 男性的: 0, 中性的: 0 };
  for (const c of allClips) {
    const p = phraseOf(c.phraseId);
    for (const t of p?.tags ?? []) {
      if (t in moodCounts) { moodCounts[t]++; }
    }
  }
  const preferredMood = Object.entries(moodCounts).sort((a, b) => b[1] - a[1])[0]?.[1] > 0
    ? Object.entries(moodCounts).sort((a, b) => b[1] - a[1])[0][0]
    : null;

  // Target span at this position
  const gapRoom = !replaceClip && later ? Math.max(0, later.start - 0.65 - refTime) : null;
  const slotSpan = replaceClip
    ? Math.max(0.5, replaceClip.end - replaceClip.start)
    : (gapRoom !== null && gapRoom >= 0.9 && gapRoom <= barSec * 3.5 ? gapRoom : barSec * 2 - 0.75);

  const nearKime = (song.accents ?? []).some((a) => (a.kime || a.kind === "kime") && a.t >= refTime - 0.25 && a.t <= refTime + slotSpan + 0.4);
  const isSectionEnd = section ? refTime + slotSpan >= section.end - barSec * 1.1 : false;
  const narrowGap = gapRoom !== null && gapRoom > 0.7 && gapRoom < barSec * 1.55;
  const windowAccents = (song.accents ?? []).filter((a) => a.t >= refTime - 0.1 && a.t <= refTime + Math.max(slotSpan, 4.5) + 0.2);

  let rawMax = -99;
  let rawMin = 99;
  const rawList: { item: Phrase; raw: number; reason: string }[] = [];

  for (const item of items) {
    const tags = new Set(item.tags ?? []);
    const desc = item.description ?? "";
    const nameLow = item.name.toLowerCase();

    const catIdx = flow.categories.indexOf(item.category);
    let catScore = catIdx >= 0 ? (flow.categories.length - catIdx) / flow.categories.length : 0;
    if (catIdx < 0 && item.category === "pose" && (nearKime || isSectionEnd || narrowGap)) {
      catScore = 0.75;
    }

    const energyScore = Math.max(0, 1 - Math.abs(item.energy - targetEnergy) / 4);

    // Accent alignment
    let alignScore = 0.5;
    if (item.accents?.length && windowAccents.length) {
      const estSpan = Math.max(0.5, Math.min(slotSpan, item.duration / 0.8));
      const spd = item.duration / estSpan;
      let sum = 0;
      for (const acc of item.accents) {
        const when = refTime + acc / spd;
        let bestErr = 99;
        let bestBoost = 1;
        for (const sa of windowAccents) {
          const d = Math.abs(sa.t - when);
          if (d < bestErr) {
            bestErr = d;
            bestBoost = sa.kime || sa.kind === "kime" ? 1.25 : sa.onBeat ? 1.1 : 1.0;
          }
        }
        const s = bestErr < 0.05 ? 1 : Math.max(0, 1 - (bestErr - 0.05) / 0.2);
        sum += Math.min(1, s * bestBoost);
      }
      alignScore = sum / item.accents.length;
    }

    // Novelty
    let novelty = 1.0;
    if (replaceClip && item.id === replaceClip.phraseId) { novelty = -0.5; }
    else if (item.id === lastId) { novelty = 0.0; }
    else if (recentIds.has(item.id)) { novelty = 0.3; }
    else if (recentStems.slice(-2).includes(baseStem(item.name))) { novelty = 0.4; }

    let raw = 0.34 * catScore + 0.24 * energyScore + 0.24 * alignScore + 0.18 * novelty;
    let reason = `${KIND_LABELS[kind] ?? kind}向き`;

    const isOutro = tags.has("アウトロ") || nameLow.includes("outro") || desc.includes("動きを収めて");
    const isIntro = tags.has("イントロ") || nameLow.includes("intro");

    if (isOutro) {
      if (kind === "outro") { raw += 0.3; reason = "アウトロ最適"; }
      else { raw -= 0.45; }
    }
    if (isIntro) {
      if (kind === "intro") { raw += 0.28; reason = "イントロ最適"; }
      else if (kind === "chorus" || kind === "build") { raw -= 0.28; }
      else { raw -= 0.14; }
    }
    if (kind === "build" && (tags.has("ビルドアップ") || tags.has("溜め") || nameLow.includes("build"))) {
      raw += 0.24;
      reason = "Bメロ・溜め向き";
    }
    if (kind === "chorus" && (tags.has("サビ") || desc.includes("コンボ") || tags.has("爆発") || nameLow.includes("combo"))) {
      raw += 0.20;
      reason = "サビ見せ場向き";
    }
    if ((nearKime || isSectionEnd) && hasEndingPose(item) && !isOutro && !isIntro) {
      raw += 0.20;
      reason = nearKime ? "キメ音に一致" : "区間締めに最適";
    }
    if (narrowGap && item.bars === 1) {
      raw += 0.22;
      reason = "隙間に収まる1小節";
    }

    // Duration / tempo fit
    const speedAtSlot = item.duration / Math.max(0.4, slotSpan);
    if (speedAtSlot >= 0.78 && speedAtSlot <= 1.38) {
      raw += 0.12;
      if (alignScore >= 0.72 && reason.endsWith("向き")) { reason = `${KIND_LABELS[kind] ?? kind}・音一致`; }
    } else if (speedAtSlot > 1.85 || speedAtSlot < 0.5) {
      raw -= 0.12;
    }

    if (preferredMood) {
      if (tags.has(preferredMood)) { raw += 0.08; }
      else if ((preferredMood === "女性的" && tags.has("男性的")) || (preferredMood === "男性的" && tags.has("女性的"))) { raw -= 0.10; }
    }

    if (raw > rawMax) { rawMax = raw; }
    if (raw < rawMin) { rawMin = raw; }
    rawList.push({ item, raw, reason });
  }

  const span = Math.max(0.25, rawMax - rawMin);
  for (const entry of rawList) {
    const norm = Math.max(0, Math.min(1, (entry.raw - rawMin) / span));
    const percent = Math.round(52 + norm * 46);
    result.set(entry.item.id, { score: entry.raw, percent, reason: entry.reason });
  }
  void clipDuration;
  return result;
}
