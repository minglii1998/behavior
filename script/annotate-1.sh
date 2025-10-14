cd ..
annotators="gpt-4.1"
for annotator in $annotators; do
    for response_model in "gemini2.0flash" "gemini2.5flash" "gpt4o" "o1mini"; do
        python -m method.gpt_all --gpt_model $annotator --llm_model $response_model
    done
done
