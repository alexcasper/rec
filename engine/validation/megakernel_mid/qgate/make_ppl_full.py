"""rec-kcb: freeze the full-config ppl set (eval_set_full_v1.json): 8 real-text items x 192 tokens, Flint tokenizer.
192 tokens so the split-K branch (GEPA_ATTN_SPLIT_MIN=64) runs on 2/3 of the teacher-forced positions.
    python make_ppl_full.py /home/dawn/rec-data/flint-int4 eval_set_full_v1.json"""
import hashlib
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

N_TOK = 192
TEXTS = {
    "gettysburg": (
        "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, "
        "and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing "
        "whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great "
        "battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who "
        "here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. "
        "But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The brave "
        "men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The "
        "world will little note, nor long remember what we say here, but it can never forget what they did here. It is for "
        "us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so "
        "nobly advanced."),
    "moby_dick": (
        "Call me Ishmael. Some years ago -- never mind how long precisely -- having little or no money in my purse, and "
        "nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the "
        "world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing "
        "grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily "
        "pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my "
        "hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately "
        "stepping into the street, and methodically knocking people's hats off -- then, I account it high time to get to "
        "sea as soon as I can. This is my substitute for pistol and ball. With a philosophical flourish Cato throws himself "
        "upon his sword; I quietly take to the ship. There is nothing surprising in this. If they but knew it, almost all men "
        "in their degree, some time or other, cherish very nearly the same feelings towards the ocean with me."),
    "pride_prejudice": (
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a "
        "wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this "
        "truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of "
        "some one or other of their daughters. \"My dear Mr. Bennet,\" said his lady to him one day, \"have you heard that "
        "Netherfield Park is let at last?\" Mr. Bennet replied that he had not. \"But it is,\" returned she; \"for Mrs. Long "
        "has just been here, and she told me all about it.\" Mr. Bennet made no answer. \"Do you not want to know who has "
        "taken it?\" cried his wife impatiently. \"You want to tell me, and I have no objection to hearing it.\" This was "
        "invitation enough. \"Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large "
        "fortune from the north of England; that he came down on Monday in a chaise and four to see the place.\""),
    "constitution": (
        "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic "
        "Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to "
        "ourselves and our Posterity, do ordain and establish this Constitution for the United States of America. Article "
        "I. Section 1. All legislative Powers herein granted shall be vested in a Congress of the United States, which "
        "shall consist of a Senate and House of Representatives. Section 2. The House of Representatives shall be composed "
        "of Members chosen every second Year by the People of the several States, and the Electors in each State shall "
        "have the Qualifications requisite for Electors of the most numerous Branch of the State Legislature. No Person "
        "shall be a Representative who shall not have attained to the Age of twenty five Years, and been seven Years a "
        "Citizen of the United States, and who shall not, when elected, be an Inhabitant of that State in which he shall be "
        "chosen. Representatives and direct Taxes shall be apportioned among the several States which may be included "
        "within this Union, according to their respective Numbers."),
    "python_code": (
        "import heapq\nfrom collections import defaultdict\n\n\ndef dijkstra(graph, source):\n"
        "    \"\"\"Return the shortest distance from source to every reachable node.\n\n"
        "    graph maps a node to a list of (neighbour, weight) pairs with non-negative weights.\n    \"\"\"\n"
        "    dist = defaultdict(lambda: float(\"inf\"))\n    dist[source] = 0\n    heap = [(0, source)]\n"
        "    while heap:\n        d, node = heapq.heappop(heap)\n        if d > dist[node]:\n            continue\n"
        "        for neighbour, weight in graph.get(node, []):\n            nd = d + weight\n"
        "            if nd < dist[neighbour]:\n                dist[neighbour] = nd\n"
        "                heapq.heappush(heap, (nd, neighbour))\n    return dict(dist)\n\n\n"
        "if __name__ == \"__main__\":\n    g = {\"a\": [(\"b\", 1), (\"c\", 4)], \"b\": [(\"c\", 2), (\"d\", 5)], "
        "\"c\": [(\"d\", 1)]}\n    print(dijkstra(g, \"a\"))\n    assert dijkstra(g, \"a\")[\"d\"] == 4\n"),
    "gpu_tech": (
        "A graphics processor hides memory latency by keeping many wavefronts resident on each compute unit. When one "
        "wavefront stalls on a global memory load, the scheduler issues instructions from another wavefront that is ready, "
        "so the arithmetic units stay busy. The number of wavefronts that can be resident at once is limited by the "
        "register file, the local data share, and a hardware cap per unit; a kernel that uses many vector registers per "
        "lane therefore runs fewer wavefronts and has less latency hiding. For a decode step that streams quantized "
        "weights, throughput is usually bounded by memory bandwidth rather than arithmetic: each weight byte is read once "
        "per token, and the dequantization work is small compared with the time to fetch it. Fusing all layers into one "
        "persistent kernel removes launch overhead, but it introduces grid-wide barriers between phases, and every "
        "barrier costs a few microseconds while the slowest workgroup catches up. Reducing the number of barriers, or "
        "overlapping the next phase's weight fetch with the tail of the current one, is often worth more than any change "
        "to the arithmetic inside a single phase."),
    "math_text": (
        "Let f be a continuous function on the closed interval [a, b] and differentiable on the open interval (a, b). The "
        "mean value theorem states that there exists a point c in (a, b) such that f'(c) = (f(b) - f(a)) / (b - a). To "
        "prove it, define g(x) = f(x) - f(a) - (x - a)(f(b) - f(a)) / (b - a). Then g is continuous on [a, b], "
        "differentiable on (a, b), and g(a) = g(b) = 0. By Rolle's theorem there is a point c with g'(c) = 0, which is "
        "exactly the claimed identity. A useful corollary follows at once: if f'(x) = 0 for every x in (a, b), then f is "
        "constant on [a, b], because for any two points the difference f(y) - f(x) equals f'(c)(y - x) for some c between "
        "them, and that product vanishes. Similarly, a positive derivative everywhere forces f to be strictly increasing."),
    "dialogue": (
        "User: Can you explain why my build fails only on the release configuration?\nAssistant: Release builds enable "
        "optimizations that debug builds skip, so code that relies on undefined behaviour often works in debug and breaks "
        "in release. Common causes are reading an uninitialized variable, a data race between threads, or an out-of-bounds "
        "array access that happens to land on harmless memory in one layout but not the other.\nUser: How do I find which "
        "one it is?\nAssistant: Start by compiling the release configuration with sanitizers enabled: address sanitizer "
        "for memory errors and thread sanitizer for races. If the failure is timing dependent, run it many times in a loop "
        "and record whether the output changes between runs. Then bisect: turn individual optimization flags off until the "
        "failure disappears, and look at the code the last flag changes.\nUser: Thanks, I will try the sanitizers first.\nAssistant: Good plan. Keep the failing input and the exact "
        "compiler version written down, so that once you have a fix you can confirm it against the same configuration "
        "that failed, rather than a nearby one that happens to pass."),
}


def main() -> int:
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    tok = Tokenizer.from_file(str(ckpt / "tokenizer.json"))
    items = []
    for k, text in TEXTS.items():
        ids = tok.encode(text, add_special_tokens=False).ids
        if len(ids) < N_TOK:
            raise SystemExit(f"{k}: only {len(ids)} tokens < {N_TOK}")
        items.append({"id": f"full/{k}", "tokens": ids[:N_TOK]})
    tsha = hashlib.sha256((ckpt / "tokenizer.json").read_bytes()).hexdigest()
    body = {"schema": "rec-qgate-ppl-full/1", "version": 1, "n_tok": N_TOK, "tokenizer_sha256": tsha,
            "ckpt_hint": str(ckpt), "ppl_items": items}
    body["sha256"] = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
    out.write_text(json.dumps(body, indent=1) + "\n")
    print(out, body["sha256"], [len(i["tokens"]) for i in items])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
