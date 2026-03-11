from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Union

from google import genai
from google.genai import types


@dataclass
class Settings:
    """
    作用：集中管理“最小必要配置”，后续我们会在这里继续加：文件上传、结构化输出、重试、链式步骤等。
    """
    model: str = "gemini-3.1-pro-preview" 
    temperature: float = 0.2
    max_output_tokens: int = 65536

    # keepalive_thoughts=True 时：请求返回 thought summary 作为“心跳”，防止长时间无输出导致连接空闲超时
    # 注意：我们不会保存 thought summary，只会丢弃它
    keepalive_thoughts: bool = True
    debug_stream: bool = False  # 打开后会在流结束时打印 finish_reason/字符数，帮助定位截断原因

    # 可选：如果你需要走代理，把它填上（也可直接在 shell export HTTPS_PROXY/HTTP_PROXY）
    # proxy: Optional[str] = "socks5://127.0.0.1:7891"
    proxy: Optional[str] = None


GeminiContents = Union[str, Sequence[Any]]
"""
Gemini Python SDK 的 generate_content / generate_content_stream 的 contents 参数既可以是纯文本 str，
也可以是一个“混合列表”，例如：
  ["Please review this PDF.", uploaded_file]
其中 uploaded_file 来自 client.files.upload(...).
（官方 Files API 示例就是把 myfile 直接放进 contents 列表里。）:contentReference[oaicite:0]{index=0}
"""

try:
    import importlib.util
except ImportError:
    pass

def load_prompts_module(path: Union[str, Path]):
    """
    动态加载 prompts 模块
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"找不到 prompts 文件：{p}")
    
    spec = importlib.util.spec_from_file_location("dynamic_prompts", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 prompts 文件：{p}")
    
    module = importlib.util.module_from_spec(spec)
    # sys.modules["dynamic_prompts"] = module # 可选，避免污染全局命名空间可不加
    spec.loader.exec_module(module)
    return module

def write_text(path: Union[str, Path], text: str) -> None:
    """
    干什么用：
    - 把每一步模型输出落盘成 .md，方便你人工检查与回溯。

    设计点：
    - 自动创建父目录
    - 统一 UTF-8 编码
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def load_api_key(explicit: Optional[str] = None) -> str:
    """
    干什么用：
    - 统一从（参数/环境变量）获取 API Key
    - 让你后续写 agent 时，不会每个文件都重复读环境变量

    怎么写：
    - 先用 explicit（如果传入）
    - 否则读 GEMINI_API_KEY，再读 GOOGLE_API_KEY
    """
    if explicit and explicit.strip():
        return explicit.strip()

    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.getenv(name)
        if v and v.strip():
            return v.strip()

    raise RuntimeError("缺少 API Key：请设置 GEMINI_API_KEY 或 GOOGLE_API_KEY")


def apply_proxy_if_needed(settings: Settings) -> None:
    """
    干什么用：
    - 如果你需要代理：在创建 client 之前设置 HTTPS_PROXY / HTTP_PROXY 环境变量
    - 这是官方 SDK 文档推荐的方式（httpx/aiohttp 会从环境变量读取代理）

    怎么写：
    - 只要 settings.proxy 有值，就写入两个环境变量
    """
    if settings.proxy:
        os.environ["HTTPS_PROXY"] = settings.proxy
        os.environ["HTTP_PROXY"] = settings.proxy


def make_client(api_key: str, settings: Settings) -> genai.Client:
    """
    干什么用：
    - 创建 genai.Client（这是所有 API 调用的入口）

    怎么写：
    - 先设置代理（如果需要）
    - 然后 client = genai.Client(api_key=api_key)
    - 注意：这里“不会”显式写 http2=false（按你的要求，保持默认）
    """
    apply_proxy_if_needed(settings)
    return genai.Client(
        api_key=api_key,
        http_options={'timeout': 600000.0}
        
    )


def upload_file(client: genai.Client, file_path: str) -> Any:
    """
    干什么用：
    - 用 Files API 把本地文件（这里主要是 PDF）上传到 Gemini
    - 返回的对象可以直接塞进 contents=[..., uploaded_file] 里发给模型:contentReference[oaicite:1]{index=1}

    备注：
    - 文档里提到：PDF 单文件上限为 50 MB（超了建议你改走分段/文本抽取/压缩）。:contentReference[oaicite:2]{index=2}
    """
    p = Path(file_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"找不到文件：{p}")

    # 只做一个“善意的”本地检查，避免你把超大 PDF 直接打到 API 上
    size_mb = p.stat().st_size / (1024 * 1024)
    if p.suffix.lower() == ".pdf" and size_mb > 50:
        raise ValueError(f"PDF 太大：{size_mb:.1f} MB（文档建议 PDF ≤ 50 MB）")

    return client.files.upload(file=str(p))


def make_config(settings: Settings) -> types.GenerateContentConfig:
    """
    干什么用：
    - 构造生成配置 GenerateContentConfig（温度、max tokens、可选 thinking）

    怎么写：
    - keepalive_thoughts=True 时，开启 include_thoughts，用 thought summary 当心跳
    - keepalive_thoughts=False 时，不带 thinking_config（最纯净）
    """
    thinking_cfg = None
    if settings.keepalive_thoughts:
        thinking_cfg = types.ThinkingConfig(include_thoughts=True)

    return types.GenerateContentConfig(
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        thinking_config=thinking_cfg,
    )



def stream_answer_text(
    client: genai.Client,
    settings: Settings, 
    contents: GeminiContents,
    system_instruction: str = "You are a helpful assistant.",
) -> Iterator[str]:
    """
    干什么用：
    - 以“流式”方式获取模型输出，并且只产出“答案文本”（不产出 thought summary）
    - 你后面的审稿 agent 会把每一步输出写入 md，核心就是靠它提供稳定文本流

    怎么写（关键点）：
    1) 用 client.models.generate_content_stream 发起流式请求（官方示例就是这样拿 chunk.text）
    2) 如果开启 include_thoughts：优先按 parts 分流（part.thought=True 的丢弃，仅打印 '.' 作为心跳）
       否则：直接使用 chunk.text 作为答案增量
    """
    cfg = make_config(settings)
    cfg.system_instruction = system_instruction  # system_instruction 放在 config 里（官方 SDK 支持）

    for chunk in client.models.generate_content_stream(
        model=settings.model,
        contents=contents,
        config=cfg,
    ):
        # 开启 keepalive_thoughts 时，按 parts 分流最稳（官方 thinking 文档就是用 part.thought 区分）
        if settings.keepalive_thoughts:
            cands = getattr(chunk, "candidates", None)
            if cands and cands[0].content and cands[0].content.parts:
                for part in cands[0].content.parts:
                    text = getattr(part, "text", None)
                    if not text:
                        continue
                    if getattr(part, "thought", False):
                        print(".", end="", flush=True)  # 心跳：不写入答案
                        continue
                    yield text
                continue  # 这个 chunk 已处理完

        # 默认路径：官方推荐的流式方式是 chunk.text
        text2 = getattr(chunk, "text", None)
        if text2:
            yield text2
        else:
            # 即使没有文本，也给个心跳，避免“看起来卡住”
            if settings.keepalive_thoughts:
                print(".", end="", flush=True)


def generate_text(
    client: genai.Client,
    settings: Settings,
    contents: GeminiContents,
    system_instruction: str = "You are a helpful assistant.",
) -> str:
    """
    干什么用：
    - 把 stream_answer_text() 产出的片段拼起来，得到最终完整字符串
    - 后续“审稿五步链式”每一步都会用它拿到完整文本

    怎么写：
    - 用 list 收集分片，最后 ''.join
    """
    pieces = []
    for s in stream_answer_text(client, settings, contents, system_instruction=system_instruction):
        pieces.append(s)
        print(s, end="", flush=True)  # 同步打印，方便你在控制台观察流式输出
    if settings.keepalive_thoughts:
        print()  # 让心跳点点点结束后换行
    return "".join(pieces)

class ReviewAgent:
    """
    这个类把“上传 PDF + 5 轮 prompt”封装起来，跑一遍就能产出 5 个阶段文件。

    你后续继续扩展时（比如加重试、加结构化输出、加中间缓存），也都放在这里最合适。
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        api_key: Optional[str] = None,
        prompts_path: Optional[str] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.api_key = load_api_key(api_key)
        self.client = make_client(self.api_key, self.settings)
        
        # 默认加载 prompts_comprehensive.py
        if prompts_path is None:
            prompts_path = str(Path(__file__).parent / "prompts_comprehensive.py")
            
        self.prompts_mod = load_prompts_module(prompts_path)
        self.system_prompt = getattr(self.prompts_mod, "SYSTEM_PROMPT", "You are a helpful assistant.")

    # def _call_step(self, step_title: str, prompt_text: str, uploaded_pdf: Any) -> str:
    #     """
    #     单步调用的统一入口。

    #     关键点：Gemini 的“文件理解”需要把 uploaded_pdf 对象放进 contents 列表里：
    #         contents = [prompt_text, uploaded_pdf]
    #     """
    #     print("\n" + "=" * 70)
    #     print(step_title)
    #     print("=" * 70)

    #     contents = [prompt_text, uploaded_pdf]
    #     return generate_text(
    #         self.client,
    #         self.settings,
    #         contents=contents,
    #         system_instruction=self.system_prompt,
    #     )
    def _call_step(self, step_title: str, prompt_text: str, uploaded_pdf: Any) -> str:
        """
        单步调用的统一入口（已增加抗高并发的指数退避重试机制）。
        """
        import time # 如果文件头部没引，可以在这里引
        
        print("\n" + "=" * 70)
        print(step_title)
        print("=" * 70)

        contents = [prompt_text, uploaded_pdf]
        
        max_retries = 3  # 最大重试次数
        for attempt in range(max_retries):
            try:
                return generate_text(
                    self.client,
                    self.settings,
                    contents=contents,
                    system_instruction=self.system_prompt,
                )
            except Exception as e:
                error_msg = str(e)
                # 捕获 503(服务器挤爆) 或 429(触发频率限制)
                if "503" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        # 指数退避：第一次等 10秒，第二次等 20秒...
                        wait_time = (2 ** attempt) * 10
                        print(f"\n\n[警告] 服务器当前拥挤 (503/429)。正为你排队，等待 {wait_time} 秒后进行第 {attempt + 2} 次重试...")
                        time.sleep(wait_time)
                    else:
                        print(f"\n\n[致命] 连续 {max_retries} 次重试失败，Google 服务器可能挂了。")
                        raise e
                else:
                    # 如果是其他错误（比如代码写错），直接抛出，不重试
                    raise e

    def run(
        self,
        pdf_path: str,
        paper_title: Optional[str] = None,
        outdir: Optional[Union[str, Path]] = "outputs",
    ) -> dict:
        """
        跑完整 5 轮。

        返回一个 dict，包含 5 个阶段的字符串结果：
          - review_v1, audit_v1, review_v2, audit_v2, review_final
        """
        pdf_p = Path(pdf_path).expanduser().resolve()
        title = paper_title or pdf_p.stem

        # 1) 上传 PDF（只做一次）
        uploaded = upload_file(self.client, str(pdf_p))

        # 2) Step 1: 初稿审稿
        review_v1 = self._call_step(
            "Step 1/5 - Initial Review",
            self.prompts_mod.PROMPT_1.format(paper_title=title),
            uploaded,
        )

        # 3) Step 2: 自审（找幻觉/弱论断）
        audit_v1 = self._call_step(
            "Step 2/5 - Self-Audit (Hallucination & Weak Claims)",
            self.prompts_mod.PROMPT_2.format(paper_title=title, review_v1=review_v1),
            uploaded,
        )

        # 4) Step 3: 修订后的审稿意见
        review_v2 = self._call_step(
            "Step 3/5 - Revised Review",
            self.prompts_mod.PROMPT_3.format(paper_title=title, review_v1=review_v1, audit_v1=audit_v1),
            uploaded,
        )

        # 5) Step 4: 第二轮覆盖自审（含附录）
        audit_v2 = self._call_step(
            "Step 4/5 - Coverage Audit (Include Appendices)",
            self.prompts_mod.PROMPT_4.format(paper_title=title, review_v2=review_v2),
            uploaded,
        )

        # 6) Step 5: 最终可核查审稿意见
        review_final = self._call_step(
            "Step 5/5 - Final Verified Review",
            self.prompts_mod.PROMPT_5.format(paper_title=title, review_v2=review_v2, audit_v2=audit_v2),
            uploaded,
        )

        results = {
            "paper_title": title,
            "review_v1": review_v1,
            "audit_v1": audit_v1,
            "review_v2": review_v2,
            "audit_v2": audit_v2,
            "review_final": review_final,
        }

        # 7) 可选：落盘
        if outdir:
            out = Path(outdir) / pdf_p.stem
            out.mkdir(parents=True, exist_ok=True)
            write_text(out / "review_v1.md", review_v1)
            write_text(out / "audit_v1.md", audit_v1)
            write_text(out / "review_v2.md", review_v2)
            write_text(out / "audit_v2.md", audit_v2)
            write_text(out / "review_final.md", review_final)
            write_text(out / "all.json", str(results))  # 最简：先直接 str；你后续可换 json.dumps

            print(f"\n[OK] Review pipeline done. Outputs in: {out.resolve()}")

        return results


def smoke_test(model: Optional[str] = None) -> None:
    """
    干什么用：
    - 从零开始的第一步：先确认“Key / 网络 / SDK / 流式”都能跑通
    - 跑通后我们再进入下一步：文件上传（PDF）+ 多步审稿链

    怎么写：
    - 创建 settings
    - load_api_key -> make_client -> generate_text
    """
    settings = Settings()
    if model:
        settings.model = model
    api_key = load_api_key()
    client = make_client(api_key, settings)

    prompt = "用一句话解释奥卡姆剃刀，并举一个生活中的例子。"
    _ = generate_text(
        client,
        settings,
        contents=prompt,
        system_instruction="你是一个严谨的中文助手。",  
    )


def smoke_test_pdf(pdf_path: str, model: Optional[str] = None) -> None:
    """
    干什么用：
    - 验证：Files API 上传 + 带文件的 generate_content_stream 能正常工作
    """
    settings = Settings()
    if model:
        settings.model = model
    api_key = load_api_key()
    client = make_client(api_key, settings)

    uploaded = upload_file(client, pdf_path)
    contents = [
        "请阅读我上传的 PDF 论文，先用 5 条要点概括贡献，再列出 5 个你认为最可能存在的问题。",
        uploaded,
    ]
    _ = generate_text(
        client,
        settings,
        contents=contents,
        system_instruction="你是一个严谨的中文学术助手。",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gemini reviewer (PDF via Files API)")

    sub = parser.add_subparsers(dest="cmd")

    p_review = sub.add_parser("review", help="Run 5-step reviewer pipeline on a PDF")
    p_review.add_argument("pdf", help="Path to paper PDF")
    p_review.add_argument("--title", default=None, help="Paper title (default: PDF filename stem)")
    p_review.add_argument("--outdir", default="outputs", help="Output directory")
    p_review.add_argument("--model", default=None, help="Override model name in Settings")
    p_review.add_argument("--prompts", default=None, help="Path to prompts python file (default: src/reviewers/prompts_comprehensive.py)")

    p_smoke = sub.add_parser("smoke", help="Run basic text smoke test")
    p_smoke.add_argument("--model", default=None, help="Override model name in Settings")

    p_pdf = sub.add_parser("pdf", help="Run quick PDF smoke test (summary + issues)")
    p_pdf.add_argument("pdf", help="Path to paper PDF")
    p_pdf.add_argument("--model", default=None, help="Override model name in Settings")

    args = parser.parse_args()

    if args.cmd == "review":
        settings = Settings()
        if args.model:
            settings.model = args.model

        agent = ReviewAgent(settings=settings, prompts_path=args.prompts)
        agent.run(args.pdf, paper_title=args.title, outdir=args.outdir)

    elif args.cmd == "pdf":
        smoke_test_pdf(args.pdf, model=args.model)

    else:
        smoke_test(model=getattr(args, "model", None))