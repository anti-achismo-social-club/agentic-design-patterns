#!/bin/bash

# Array of document IDs
doc_ids=(
    "1cQ61mNpiWn6eSORmWjEjF44vN2Lpba8kyKmNwIC60ig"
    "1u2y6tY48bw8nriDUuwWEf9s8g66vyIqBKSKZDOS-n0s"
    "18Q9kfZuCTL37ztrSjLxwf8Elr5UfAiAavmnj0IqSpbU"
    "1PWhaXD_UNKgJaxYe3JBxRFRt3_B8Wm67CFxtSBQ4LkU"
    "1K5jwqB6jh20uHL0TTWxqWOxFk-dzFxRvHzrRRV79hrg"
    "1Nw6hRa7ItdLr_Tj5hF2q-OH8B_uPKb--RLn8SXZKA94"
    "1flxKGrbnF2g8yh3F-oVD5Xx7ZumId56HbFpIiPdkqLI"
    "1XVMp4RcRkoUJTVbrP2foWZX703CUJpWkrhyFU2cfUOA"
    "1ux_n8n3T4bYndOjs1DKW5ccpC802KISdy2IWnlvYbas"
    "1HXXJOQIMWowtLw4WMiSR360caDAlZPtl5dPPgvq9IT4"
    "1bE4iMljhppqGY1p48gQWtZvk6MfRuJRCiba1yRykGNE"
    "18vvNESEwHnVUREzIipuaDNCnNAREGqEfy9MQYC9wb4o"
    "1RZ5-2fykDQKOBx01pwfKkDe0GCs5ydca7xW9Q4wqS_M"
    "1asVTObtzIye0I9ypAztaeeI_sr_Hx2TORE02uUuqH_c"
    "1UHTEDCmSM1nwB-iyMoHuYzVcu_B_4KkJ2ITGGUKqo8s"
    "1e6XimYczKmhX9zpqEyxLFWPQgGuG0brp7Hic2sFl_qw"
    "10ndlCB39BWjyFRWKpcoKib4vuPD1ojD-x0-ynMaf5uw"
    "1C07AuMur6-infwE0viCp4QtAy_wWI-uceFm6MaYHQGk"
    "1ImOZcw6yeb7a-uRBMNP1VdovYfyip4IdsAcLu9yue-0"
    "1v96Oobio6xDOqbK8ejsXjmOc4Dp2uoLMo5_gfJgi-NE"
    "1H6HmUYcy5kugt5gt7Kh2Zzb8C62d5pu36RsgMNDCX24"
    "1nAN58l6JjqEJHk43126uh7xgdEblCpcbsNUHXgtBmJQ"
    "1Yt1W_hLaC6ZNgJXfT4W6NrCL4TzNVdKOX50kgpHiIq4"
    "1Gpc5af_okze1kprRLohP6-81e1KwL6HggjeLvxQyIuk"
    "1G3zOZM2ZOd0gUp5dy66FUjKMOcALh9l-JpvPxgGMm8w"
    "1qyXxGM2hNqW_qjXuBFxrEUeoYVO79BoW1ogKu1bfdCY"
    "1zeeMVTqjqRIli6G9MMWThhoQhvKqLOjJF2EHHUXLhdk"
    "1V7EKEWibOH6IhHD_PtbFZiml492-2191jDQCcTkhtTI"
    "11pma_tCoC7uZ2SFKjcR5KyIq0_ooMGSoadI6f9mxG2I"
    "151rGsiEYOkXUcNDRus_N8TxxuvjoyTDViBhzt9z0Mfw"
    "1bDRJ8mKtLTeWNC-cGD0Cr8pEJQgJHNcjqz5ekloAjaE"
    "1W4znto0a8Ikajw5a4tEyRAaB2nJPJw_iFc4w4qNnjho"
    "14q3fQ-FZmDgiughno_WLSILMWkURvUgR7mlGiFtvwd4"
    "1tVyhgwrD4fu_D_pHUrwhNxoguRG3tLc1KObXFxrxE_s"
    "1_j_OdzeUALluBUO1GkZ48DsHbbDETiM_1G4farVLPnE"
    "15MrpoJBrZIi6aEZrBJCeCvVsIoC-EecYG9EWVk3YCKw"
    "1XxQsHX3FWEP3TisQeWZwFfYw81zDwEHfHdIfjypa_0g"
)

# Base directory for downloads
base_dir="/Users/lucianfialho/Code/personal/agentic-design-patterns/downloaded_docs"

echo "Starting download of ${#doc_ids[@]} documents..."

# Download each document
for i in "${!doc_ids[@]}"; do
    doc_id="${doc_ids[$i]}"
    file_name="doc_$(printf "%02d" $((i+1)))_${doc_id}.txt"
    url="https://docs.google.com/document/d/${doc_id}/export?format=txt"

    echo "Downloading document $((i+1))/37: ${doc_id}..."

    if curl -L -s "$url" -o "${base_dir}/${file_name}"; then
        echo "Successfully downloaded: ${file_name}"
    else
        echo "Failed to download: ${doc_id}"
    fi

    # Small delay to be respectful to Google's servers
    sleep 1
done

echo "Download complete!"
echo "Files saved in: ${base_dir}"