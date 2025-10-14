cd ..
# annotators="gpt-4.1" "gpt-5" "gemini-2.5-flash"
for annotator in "gpt-4.1" "gpt-5" "gemini-2.5-flash"; do
    for response_model in "gemini2.0flash"; do
        python -m method.gpt_all --gpt_model $annotator --llm_model $response_model
    done
done
