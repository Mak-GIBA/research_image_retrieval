#!/bin/bash

# Table 1 全モデル学習実行スクリプト
# All Table 1 Models Training Master Script

echo "Table 1 Models Training Master Script"
echo "====================================="
echo ""

# 利用可能なモデル一覧
MODELS=(
    "gem_r50"
    "gem_r101" 
    "delg_r50"
    "delg_r101"
    "token_r50"
    "token_r101"
    "how_vlad_r50"
    "how_vlad_r101"
    "how_asmk_r50"
    "how_asmk_r101"
    "senet_g2_50"
    "senet_g2_101"
    "sosnet_r50"
    "sosnet_r101"
    "spoc_r50"
    "spoc_r101"
)

# 使用方法を表示
show_usage() {
    echo "Usage: $0 [MODEL_NAME|all|list]"
    echo ""
    echo "Options:"
    echo "  MODEL_NAME  - Train specific model (e.g., gem_r50)"
    echo "  all         - Train all models sequentially"
    echo "  list        - List all available models"
    echo ""
    echo "Available models:"
    for model in "${MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "Examples:"
    echo "  $0 gem_r50              # Train GeM R50 model"
    echo "  $0 all                  # Train all models"
    echo "  $0 list                 # List available models"
}

# モデル一覧を表示
list_models() {
    echo "Available Table 1 Models:"
    echo "========================"
    for i in "${!MODELS[@]}"; do
        model="${MODELS[$i]}"
        script_file="training_scripts/${model}_training.sh"
        if [ -f "$script_file" ]; then
            status="✓ Available"
        else
            status="✗ Script not found"
        fi
        printf "%2d. %-20s %s\n" $((i+1)) "$model" "$status"
    done
}

# 特定のモデルを学習
train_model() {
    local model_name="$1"
    local script_file="training_scripts/${model_name}_training.sh"
    
    if [ ! -f "$script_file" ]; then
        echo "Error: Training script not found for model '$model_name'"
        echo "Expected: $script_file"
        return 1
    fi
    
    echo "Starting training for model: $model_name"
    echo "Script: $script_file"
    echo "----------------------------------------"
    
    # 実行スクリプトを呼び出し
    bash "$script_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ Training completed successfully for $model_name"
    else
        echo "✗ Training failed for $model_name"
        return 1
    fi
}

# 全モデルを順次学習
train_all_models() {
    echo "Starting training for all Table 1 models..."
    echo "This will take a very long time!"
    echo ""
    
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Training cancelled."
        return 1
    fi
    
    local failed_models=()
    
    for model in "${MODELS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Training model: $model"
        echo "=========================================="
        
        if train_model "$model"; then
            echo "✓ $model training completed"
        else
            echo "✗ $model training failed"
            failed_models+=("$model")
        fi
        
        # 次のモデルまで少し待機
        sleep 5
    done
    
    # 結果サマリー
    echo ""
    echo "=========================================="
    echo "Training Summary"
    echo "=========================================="
    echo "Total models: ${#MODELS[@]}"
    echo "Successful: $((${#MODELS[@]} - ${#failed_models[@]}))"
    echo "Failed: ${#failed_models[@]}"
    
    if [ ${#failed_models[@]} -gt 0 ]; then
        echo ""
        echo "Failed models:"
        for model in "${failed_models[@]}"; do
            echo "  - $model"
        done
    fi
}

# メイン処理
main() {
    if [ $# -eq 0 ]; then
        show_usage
        return 1
    fi
    
    case "$1" in
        "list")
            list_models
            ;;
        "all")
            train_all_models
            ;;
        *)
            # 特定のモデル名が指定された場合
            if [[ " ${MODELS[@]} " =~ " $1 " ]]; then
                train_model "$1"
            else
                echo "Error: Unknown model '$1'"
                echo ""
                show_usage
                return 1
            fi
            ;;
    esac
}

# スクリプト実行
main "$@"

