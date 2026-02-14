#!/bin/bash
# ==========================================
# Download DomainNet Dataset
# ==========================================
# Downloads all 6 domains of DomainNet (cleaned version)
# Total size: ~18.5 GB
# Domains: clipart, infograph, painting, quickdraw, real, sketch
# 345 classes per domain

DATA_DIR="${1:-/umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/data/DomainNet}"

echo "============================================"
echo "Downloading DomainNet Dataset"
echo "Target directory: ${DATA_DIR}"
echo "============================================"

mkdir -p "${DATA_DIR}"

# DomainNet download URLs (from ai.bu.edu / visda challenge)
DOMAINS=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")
BASE_URL="http://csr.bu.edu/ftp/visda/2019/multi-source"

# Download and extract each domain
for DOMAIN in "${DOMAINS[@]}"; do
    ZIP_FILE="${DATA_DIR}/${DOMAIN}.zip"
    DOMAIN_DIR="${DATA_DIR}/${DOMAIN}"
    
    # Skip if already extracted
    if [ -d "${DOMAIN_DIR}" ] && [ "$(ls -A ${DOMAIN_DIR} 2>/dev/null)" ]; then
        echo "[SKIP] ${DOMAIN} already exists at ${DOMAIN_DIR}"
        continue
    fi
    
    echo ""
    echo ">>> Downloading ${DOMAIN}..."
    
    # Determine URL — real and sketch have different paths in some mirrors
    if [ "${DOMAIN}" == "real" ] || [ "${DOMAIN}" == "sketch" ]; then
        URL="${BASE_URL}/${DOMAIN}.zip"
    else
        URL="${BASE_URL}/groundtruth/${DOMAIN}.zip"
    fi
    
    # Download
    if [ ! -f "${ZIP_FILE}" ]; then
        echo "    URL: ${URL}"
        wget -q --show-progress -O "${ZIP_FILE}" "${URL}"
        
        # Check if download succeeded
        if [ $? -ne 0 ] || [ ! -s "${ZIP_FILE}" ]; then
            echo "    [WARN] Primary URL failed, trying alternate..."
            # Try alternate URL pattern
            ALT_URL="http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/${DOMAIN}.zip"
            wget -q --show-progress -O "${ZIP_FILE}" "${ALT_URL}"
            
            if [ $? -ne 0 ] || [ ! -s "${ZIP_FILE}" ]; then
                echo "    [ERROR] Failed to download ${DOMAIN}. Skipping."
                rm -f "${ZIP_FILE}"
                continue
            fi
        fi
        echo "    Download complete."
    else
        echo "    [CACHE] Using existing zip: ${ZIP_FILE}"
    fi
    
    # Extract
    echo "    Extracting ${DOMAIN}..."
    unzip -q -o "${ZIP_FILE}" -d "${DATA_DIR}"
    
    if [ $? -eq 0 ]; then
        echo "    Extraction complete."
        # Remove zip to save space
        # rm -f "${ZIP_FILE}"
        # echo "    Removed zip file."
    else
        echo "    [ERROR] Failed to extract ${DOMAIN}."
    fi
done

# Verify
echo ""
echo "============================================"
echo "Verification"
echo "============================================"
for DOMAIN in "${DOMAINS[@]}"; do
    DOMAIN_DIR="${DATA_DIR}/${DOMAIN}"
    if [ -d "${DOMAIN_DIR}" ]; then
        N_CLASSES=$(find "${DOMAIN_DIR}" -maxdepth 1 -type d | wc -l)
        N_CLASSES=$((N_CLASSES - 1))  # subtract the domain dir itself
        N_IMAGES=$(find "${DOMAIN_DIR}" -name "*.jpg" -o -name "*.png" | wc -l)
        echo "  ${DOMAIN}: ${N_CLASSES} classes, ${N_IMAGES} images"
    else
        echo "  ${DOMAIN}: NOT FOUND"
    fi
done

echo ""
echo "============================================"
echo "DomainNet download complete!"
echo "Data directory: ${DATA_DIR}"
echo "============================================"
