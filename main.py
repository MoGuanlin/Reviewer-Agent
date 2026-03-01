import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env
load_dotenv()

# Add src to path to allow running as script from root
sys.path.append(str(Path(__file__).parent))

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
                        help="Output directory. If not specified, defaults to 'outputs/<model_sanitized>'.")
    parser.add_argument("--proxy", default=None, help="Proxy URL (e.g. http://127.0.0.1:7890).")
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
    
    # Auto-detect provider
    if provider == "auto":
        model_lower = model.lower()
        if "gemini" in model_lower:
            provider = "gemini"
        elif "gpt" in model_lower or "openai" in model_lower or "deepseek" in model_lower or "anthropic" in model_lower or "claude" in model_lower:
            # Assume other models go via OpenRouter/GPT interface
            provider = "gpt"
        else:
            # Fallback to GPT (OpenRouter) for unknown models as it's more generic
            print(f"Warning: Unknown model '{model}'. Defaulting to 'gpt' provider (OpenRouter compatible).")
            provider = "gpt"
            
    print(f"Using provider: {provider} with model: {model}")
    
    # Determine output directory
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        # Sanitize model name for directory
        sanitized_model = model.replace("/", "_").replace("-", "_").replace(":", "_")
        outdir = Path("outputs") / sanitized_model
    
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir.resolve()}")
    
    # Proxy Setup
    proxy = args.proxy
    if not proxy:
        # Check environment variables
        proxy = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
        
    if proxy:
        print(f"Using proxy: {proxy}")
    
    # Extract pages
    print(f"Extracting text from {pdf_path}...")
    try:
        pages = extract_pdf_pages(str(pdf_path))
    except Exception as e:
        print(f"Error extracting PDF: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Extracted {len(pages)} pages.")
    
    try:
        if provider == "gemini":
            paper_title = pdf_path.stem 
            run_gemini_review(
                pages=pages,
                paper_title=paper_title,
                outdir=outdir,
                model=model,
                thinking_level=args.thinking_level,
                temperature=args.temperature,
                proxy=proxy,
                pdf_path=str(pdf_path)
            )
        elif provider == "gpt":
            outfile = outdir / "review.md"
            run_gpt_review(
                pages=pages,
                out_file=outfile,
                model=model,
                max_input_chars=args.max_input_chars,
                chunk_chars=args.chunk_chars,
                temperature=args.temperature,
                proxy=proxy
            )
        else:
            print(f"Unknown provider: {provider}", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"\nFATAL ERROR during review: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
