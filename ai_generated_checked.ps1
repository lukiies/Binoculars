<#
    start.ps1 - Binoculars AI-text detector launcher (4-bit, fits an 8 GB GPU)

    Run WITHOUT parameters to see this help.
    Run WITH a document path to detect how likely it was written by an AI.
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$InputPath,

    [ValidateSet('4bit', '8bit', 'none')]
    [string]$Quant = '4bit',

    [int]$MaxToken = 384   # tuned for 8 GB VRAM headroom (smaller logit/activation buffers)
)

$ErrorActionPreference = 'Stop'
$ScriptDir = $PSScriptRoot
$EnvName = 'GenAI_FA'   # conda env to use (see docs/ENVIRONMENT_SETUP.md)

# Resolve the Python interpreter to use, in priority order:
#   1) $env:BINO_PYTHON if set (explicit override)
#   2) the GenAI_FA conda env, located via 'conda info --base'
#   3) whatever 'python' is on PATH (e.g. the env is already activated)
function Resolve-Python {
    if ($env:BINO_PYTHON -and (Test-Path $env:BINO_PYTHON)) { return $env:BINO_PYTHON }
    try {
        $base = (& conda info --base) 2>$null
        if ($base) {
            $candidate = Join-Path $base "envs\$EnvName\python.exe"
            if (Test-Path $candidate) { return $candidate }
        }
    } catch {}
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}
$Py = Resolve-Python

function Show-Help {
    Write-Host ""
    Write-Host "  Binoculars - AI-generated text detector" -ForegroundColor Cyan
    Write-Host "  =======================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Estimates how likely a document was written by an AI (LLM) vs. a human,"
    Write-Host "  using the Falcon-7B observer/performer pair in 4-bit quantization so it"
    Write-Host "  fits fully into an 8 GB GPU (RTX 4060)."
    Write-Host ""
    Write-Host "  USAGE:" -ForegroundColor Yellow
    Write-Host "    .\ai_generated_checked.ps1 <path-to-document> [-Quant 4bit|8bit|none] [-MaxToken 384]"
    Write-Host ""
    Write-Host "  SUPPORTED DOCUMENT TYPES (text extraction):" -ForegroundColor Yellow
    Write-Host "    .docx        Microsoft Word (2007+). Text from paragraphs AND tables."
    Write-Host "    .pdf         Digital PDFs (text layer extracted via pypdf)."
    Write-Host "    .txt .md     Plain-text / Markdown files are read as-is (no extraction)."
    Write-Host ""
    Write-Host "    NOT supported: .doc (legacy Word 97-2003), .odt, .rtf, .pages, and"
    Write-Host "    SCANNED / image-only PDFs (no text layer - would need OCR). Convert"
    Write-Host "    those to .docx / a text-based .pdf (or paste the text into a .txt)."
    Write-Host ""
    Write-Host "  PARAMETERS:" -ForegroundColor Yellow
    Write-Host "    <path-to-document>   (required) The .docx / .pdf / .txt / .md file to analyse."
    Write-Host "    -Quant               4bit (default) | 8bit | none. 4bit fits 8 GB VRAM."
    Write-Host "    -MaxToken            Tokens per scoring window (default 384, tuned for 8 GB"
    Write-Host "                         VRAM headroom). Split into windows, scored, then"
    Write-Host "                         length-weighted. Raise it (e.g. 512) if you have more VRAM."
    Write-Host ""
    Write-Host "  VRAM: the script auto-sets PYTORCH_ALLOC_CONF=expandable_segments:True" -ForegroundColor Yellow
    Write-Host "  and uses a 384-token window so it stays comfortably under 8 GB. For maximum"
    Write-Host "  headroom, close other GPU-heavy apps (browser, Teams) before running."
    Write-Host ""
    Write-Host "  EXAMPLES:" -ForegroundColor Yellow
    Write-Host "    .\ai_generated_checked.ps1 `"C:\path\to\My Essay.docx`""
    Write-Host "    .\ai_generated_checked.ps1 .\extracted_text.txt -Quant 4bit"
    Write-Host ""
    Write-Host "  OUTPUT:" -ForegroundColor Yellow
    Write-Host "    A per-window + length-weighted aggregate Binoculars score and a verdict"
    Write-Host "    against two thresholds (0.8536 low-false-positive / 0.9015 accuracy)."
    Write-Host "    LOWER score = more AI-like. A score below the threshold reads as"
    Write-Host "    'Most likely AI-generated'; above it, 'Most likely human-generated'."
    Write-Host ""
}

# No parameters -> show help and exit.
if ([string]::IsNullOrWhiteSpace($InputPath)) {
    Show-Help
    return
}

# --- Validate environment ---
if (-not $Py -or -not (Test-Path $Py)) {
    Write-Host "ERROR: could not find a Python interpreter for the '$EnvName' env." -ForegroundColor Red
    Write-Host "Fix one of the following, then re-run:" -ForegroundColor Red
    Write-Host "  - Create the env:   see docs/ENVIRONMENT_SETUP.md" -ForegroundColor Red
    Write-Host "  - Or activate it:   conda activate $EnvName" -ForegroundColor Red
    Write-Host "  - Or set a path:    `$env:BINO_PYTHON = 'C:\path\to\python.exe'" -ForegroundColor Red
    exit 1
}

# --- Resolve input document ---
if (-not (Test-Path $InputPath)) {
    Write-Host "ERROR: file not found: $InputPath" -ForegroundColor Red
    exit 1
}
$InputFull = (Resolve-Path $InputPath).Path
$ext = [System.IO.Path]::GetExtension($InputFull).ToLowerInvariant()

# Binoculars / HF env: Xet backend stalls & corrupts on this connection.
$env:HF_HUB_DISABLE_XET = '1'
# Keep VRAM under the 8 GB ceiling: expandable segments reduce CUDA fragmentation
# and OOM risk near the limit (set before Python/torch starts). torch >= 2.6
# renamed PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF (pinned env uses 2.9.1).
$env:PYTORCH_ALLOC_CONF = 'expandable_segments:True'

# --- Route by file type to produce a plain-text file for scoring ---
switch ($ext) {
    '.docx' {
        $TextFile = Join-Path $ScriptDir 'extracted_text.txt'
        Write-Host "Extracting text from .docx ..." -ForegroundColor Cyan
        & $Py (Join-Path $ScriptDir 'scripts\extract_docx.py') $InputFull $TextFile
        if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: extraction failed." -ForegroundColor Red; exit 1 }
    }
    '.pdf' {
        $TextFile = Join-Path $ScriptDir 'extracted_text.txt'
        Write-Host "Extracting text from .pdf ..." -ForegroundColor Cyan
        & $Py (Join-Path $ScriptDir 'scripts\extract_pdf.py') $InputFull $TextFile
        if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: extraction failed." -ForegroundColor Red; exit 1 }
    }
    { $_ -in '.txt', '.md', '.markdown', '.text' } {
        $TextFile = $InputFull
    }
    default {
        Write-Host "ERROR: unsupported file type '$ext'." -ForegroundColor Red
        Write-Host "Supported: .docx / .pdf (extracted), or plain text .txt / .md (used as-is)." -ForegroundColor Red
        Write-Host "Convert legacy .doc / .odt / .rtf (and scanned PDFs) to .docx first." -ForegroundColor Red
        exit 1
    }
}

# --- Run detection ---
Write-Host "Tip: for maximum VRAM headroom on an 8 GB GPU, close other GPU-heavy apps" -ForegroundColor DarkGray
Write-Host "     (browsers, Teams, games) before running." -ForegroundColor DarkGray
Write-Host "Running Binoculars detection (quant=$Quant, max-token=$MaxToken) ..." -ForegroundColor Cyan
& $Py (Join-Path $ScriptDir 'run_detection.py') $TextFile --quant $Quant --max-token $MaxToken
exit $LASTEXITCODE
