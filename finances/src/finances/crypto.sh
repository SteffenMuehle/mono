#!/bin/bash

# Set locale to ensure dot (.) is used as the decimal separator
export LC_NUMERIC="C"

TOML_FILE="data/input/graph/crypto.md"

while IFS= read -r line; do
    # Detect the start of a section, e.g., [stellar]
    if [[ $line == \[*\] ]]; then
        full_section=$(echo "$line" | tr -d '[]')
        coin=$(echo "$full_section" | awk -F'.' '{print $NF}')
    fi

    # Extract the number (e.g., number=3081.13254248)
    if [[ $line == number* ]]; then
        number=$(echo "$line" | cut -d'=' -f2 | tr -d ' ')
    fi

    # Fetch the current price from the CoinGecko API
    if [[ $line == price* ]]; then
        price=$(curl -s "https://api.coingecko.com/api/v3/simple/price?ids=$coin&vs_currencies=eur" | jq -r ".${coin}.eur")
        
        # Calculate the total value
        total_value=$(printf "%.2f" "$(echo "$price * $number" | bc)")

        # Update the TOML file with the new price, current_amount, and target_amount
        sed -i "/^\[$full_section\]/,/^price/s/^price.*/price = $price/" "$TOML_FILE"
        sed -i "/^\[$full_section\]/,/^current_amount/s/^current_amount.*/current_amount = $total_value/" "$TOML_FILE"
        sed -i "/^\[$full_section\]/,/^target_amount/s/^target_amount.*/target_amount = $total_value/" "$TOML_FILE"

        echo "Updated $coin: total value: $total_value EUR   (price: $price EUR, number: $number)"
    fi
done < "$TOML_FILE"