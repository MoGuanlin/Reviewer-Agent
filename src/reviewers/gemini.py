# step_01_minimal_client.py
# 从“零”开始：只做一件事——用官方 Python SDK 可靠地发起一次 Gemini 文本生成（支持流式）
#
# 依赖安装：
#   pip install -U google-genai
#
# 环境变量（二选一）：
#   export GEMINI_API_KEY="你的key"
# 或
#   export GOOGLE_API_KEY="你的key"
#
# 参考（官方）：
# - 流式：for chunk in client.models.generate_content_stream(...): print(chunk.text, end="") 见 SDK 文档
# - Proxy：通过环境变量 HTTPS_PROXY / HTTP_PROXY，且建议在 client 初始化前设置

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional

from google import genai
from google.genai import types


@dataclass
class Settings:
    """
    作用：集中管理“最小必要配置”，后续我们会在这里继续加：文件上传、结构化输出、重试、链式步骤等。
    """
    model: str = "gemini-3-flash-preview"  # 先用一个常用的快速模型；后续你可换 pro 系列
    temperature: float = 0.2
    max_output_tokens: int = 4096

    # keepalive_thoughts=True 时：请求返回 thought summary 作为“心跳”，防止长时间无输出导致连接空闲超时
    # 注意：我们不会保存 thought summary，只会丢弃它
    keepalive_thoughts: bool = True

    # 可选：如果你需要走代理，把它填上（也可直接在 shell export HTTPS_PROXY/HTTP_PROXY）
    proxy: Optional[str] = None


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
    return genai.Client(api_key=api_key)


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
    prompt: str,
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
        contents=prompt,
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
    prompt: str,
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
    for s in stream_answer_text(client, settings, prompt, system_instruction=system_instruction):
        pieces.append(s)
        print(s, end="", flush=True)  # 同步打印，方便你在控制台观察流式输出
    if settings.keepalive_thoughts:
        print()  # 让心跳点点点结束后换行
    return "".join(pieces)


def smoke_test() -> None:
    """
    干什么用：
    - 从零开始的第一步：先确认“Key / 网络 / SDK / 流式”都能跑通
    - 跑通后我们再进入下一步：文件上传（PDF）+ 多步审稿链

    怎么写：
    - 创建 settings
    - load_api_key -> make_client -> generate_text
    """
    settings = Settings()
    api_key = load_api_key()
    client = make_client(api_key, settings)

    prompt = "用一句话解释奥卡姆剃刀，并举一个生活中的例子。"
    _ = generate_text(
        client,
        settings,
        prompt=prompt,
        system_instruction="你是一个严谨的中文助手。",
    )


if __name__ == "__main__":
    smoke_test()