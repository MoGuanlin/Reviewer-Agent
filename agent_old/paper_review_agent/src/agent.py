
import argparse
import os
import sys
from pathlib import Path

# Add src to path to allow running as script from root
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import extract_pdf_pages
from src.reviewers.gemini import run_gemini_review
from src.reviewers.gpt import run_gpt_review


def main():
    parser = argparse.ArgumentParser(description="Paper Review Agent")
    parser.add_argument("--pdf", required=True, help="Path to paper PDF.")
    parser.add_argument("--model", default="gemini-3.1-pro-preview", 
                        help="Model ID (e.g. gemini-3.1-pro-preview, openai/gpt-5.2-pro). Default: gemini-3.1-pro-preview")
    parser.add_argument("--provider", choices=["gemini", "gpt", "auto"], default="auto",
                        help="LLM provider (gemini or gpt). If auto, inferred from model name.")
    parser.add_argument("--outdir", default=None, 
                        help="Output directory. If not specified, defaults to model name (sanitized).")
    parser.add_argument("--proxy", default=None, help="Proxy URL (e.g. socks5://127.0.0.1:7891).")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    # Gemini specific
    parser.add_argument("--thinking_level", default="high", choices=["minimal", "low", "medium", "high"],
                        help="Gemini thinking level.")
    # GPT specific
    parser.add_argument("--max_input_chars", type=int, default=180000, help="GPT: Max chars for raw context.")
    parser.add_argument("--chunk_chars", type=int, default=35000, help="GPT: Max chars per chunk.")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)
        
    model = args.model
    provider = args.provider
    
    if provider == "auto":
        if "gemini" in model.lower():
            provider = "gemini"
        elif "gpt" in model.lower() or "openai" in model.lower():
            provider = "gpt"
        else:
            # Default fallback or error? Let's default to gpt if unknown pattern but usually it's openrouter
            # But let's look at the default. Default is gemini.
            # If user passed a custom model name that doesn't match, we might need to ask or default.
            # Let's assume GPT for unknown models as OpenRouter supports many.
            provider = "gpt"
            
    print(f"Using provider: {provider} with model: {model}")
    
    # Determine output directory
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        # Sanitize model name for directory
        sanitized_model = model.replace("/", "_").replace("-", "_")
        outdir = Path(sanitized_model)
    
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {outdir}")
    
    # Extract pages once (both reviewers need it, though format differs slightly internally handled by utils)
    # Actually utils.extract_pdf_pages returns List[Dict] which both can use (Gemini uses a wrapper in utils to format string)
    pages = extract_pdf_pages(str(pdf_path))
    
    if provider == "gemini":
        paper_title = pdf_path.stem # Simple title extraction
        run_gemini_review(
            pages=pages,
            paper_title=paper_title,
            outdir=outdir,
            model=model,
            thinking_level=args.thinking_level,
            temperature=args.temperature,
            proxy=args.proxy
        )
    elif provider == "gpt":
        # GPT reviewer outputs to a single file usually, but we can put it in outdir
        outfile = outdir / "review.md"
        run_gpt_review(
            pages=pages,
            out_file=outfile,
            model=model,
            max_input_chars=args.max_input_chars,
            chunk_chars=args.chunk_chars,
            temperature=args.temperature,
            proxy=args.proxy
        )
    else:
        print(f"Unknown provider: {provider}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
