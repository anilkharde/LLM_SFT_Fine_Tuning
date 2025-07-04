# LLM_SFT_Fine_Tuning


## **What is Supervised Fine-Tuning (SFT)?**

**SFT** is a crucial step in making language models more helpful and aligned with human preferences. Here's what happens:

###  **Key Concepts:**

1. **Instruction Following**: Train the model to follow instructions in a conversational format
2. **Response Quality**: Improve the model's ability to generate coherent, helpful responses
3. **Domain Adaptation**: Adapt the model to specific tasks or domains

### 🔧 **Technical Components:**

- **Base Model**: Pre-trained LLM (OPT-1.3B in our case)
- **Dataset**: Instruction-response pairs (EvolKit-75K)
- **Loss Function**: Cross-entropy loss on response tokens only
- **Data Collator**: `DataCollatorForCompletionOnlyLM` ensures we only train on responses

###  **Training Process:**
1. Format conversations into instruction-response pairs
2. Tokenize with special attention to response boundaries
3. Apply loss only on response tokens (not instruction tokens)
4. Monitor training with metrics like perplexity and loss



#  **Understanding TRL (Transformer Reinforcement Learning)**

##  **What is TRL?**

**TRL** is Hugging Face's library for training language models with reinforcement learning techniques. It's specifically designed for:

###  **Key TRL Components:**

1. **SFTTrainer** - Supervised Fine-Tuning (what you're using)
2. **RewardTrainer** - Training reward models 
3. **PPOTrainer** - RLHF with Proximal Policy Optimization
4. **DPOTrainer** - Direct Preference Optimization

##  **TRL Components in Your Project:**

### **1. SFTTrainer:**
- Handles instruction-following fine-tuning
- Automatically formats datasets
- Manages memory efficiently
- Integrates with HuggingFace ecosystem

### **2. SFTConfig:**
- Configuration class for SFT training
- Extends TrainingArguments with SFT-specific options
- Handles packing, sequence length, evaluation strategies

### **3. DataCollatorForCompletionOnlyLM:**
- **CRITICAL**: Only applies loss to response tokens, not instruction tokens
- Prevents the model from learning to repeat instructions
- Uses response template to identify where responses start

##  **Why Use TRL vs. Raw Transformers?**

###  **TRL Advantages:**
- **Simplified API**: Less boilerplate code
- **Optimized for LLM training**: Memory efficient, handles long sequences
- **Built-in best practices**: Proper loss masking, data formatting
- **Integration**: Works seamlessly with datasets, tokenizers, wandb


###  **With TRL:**
```python
# Clean and simple:
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_func,
    data_collator=DataCollatorForCompletionOnlyLM(...)
)
```



## **Project Completion Checklist & Next Steps**

###  **What we Accomplished:**
1. **Data Preparation**: Successfully loaded and formatted EvolKit-75K dataset
2. **Model Setup**: Configured OPT-1.3B for instruction fine-tuning
3. **Training Pipeline**: Implemented SFT with proper loss masking
4. **Evaluation**: Generated responses and saved results
5. **Monitoring**: Integrated WandB for training visualization

### 🚀 **Next Steps for Learning:**

#### **1. Advanced SFT Techniques:**
- **Parameter-Efficient Fine-tuning**: Try LoRA or QLoRA for larger models
- **Multi-task Training**: Combine different instruction datasets
- **Length Control**: Implement techniques to control response length

#### **2. Evaluation Improvements:**
- **Human Evaluation**: Get human ratings for response quality
- **Automatic Metrics**: BLEU, ROUGE, BERTScore for response quality
- **Safety Evaluation**: Check for harmful or biased outputs

#### **3. Beyond SFT - RLHF:**
- **Reward Modeling**: Train a reward model on human preferences
- **PPO Training**: Use reinforcement learning to align with human values
- **Constitutional AI**: Implement self-critique and improvement


#### **Practical Exercises:**
1. **Try Different Base Models**: Llama-2, Mistral, CodeLlama
2. **Dataset Experiments**: Compare different instruction datasets
3. **Hyperparameter Tuning**: Learning rate, batch size, sequence length


## **TRL Training Pipeline: From SFT to RLHF**

### **The Complete TRL Journey:**

```
 Pre-trained Model (GPT, OPT, Llama)
          ↓
 SFT (Supervised Fine-Tuning) ← You are here!
          ↓
 Reward Model Training
          ↓  
 RLHF (PPO Training)
          ↓
 Aligned Model (ChatGPT-style)
```

### **1. SFT Stage (Current):**
- **Goal**: Teach basic instruction following
- **Data**: Instruction-response pairs
- **Method**: Standard supervised learning
- **TRL Tools**: `SFTTrainer`, `SFTConfig`

### **2. Reward Model Stage (Next):**
- **Goal**: Learn human preferences
- **Data**: Response comparisons (A vs B)
- **Method**: Classification on preference pairs
- **TRL Tools**: `RewardTrainer`, `RewardConfig`

### **3. RLHF Stage (Advanced):**
- **Goal**: Optimize for human preferences
- **Data**: Prompts (no labels needed!)
- **Method**: Reinforcement learning (PPO)
- **TRL Tools**: `PPOTrainer`, `PPOConfig`

### **Alternative: DPO (Direct Preference Optimization):**
- **Goal**: Skip reward model, optimize directly
- **Data**: Preference pairs (like reward model)
- **Method**: Direct optimization
- **TRL Tools**: `DPOTrainer`, `DPOConfig`

### **🔧 Key TRL Features:**

1. **Memory Optimization**: Gradient checkpointing, model sharding
2. **Distributed Training**: Multi-GPU, multi-node support  
3. **Integration**: Works with Accelerate, DeepSpeed
4. **Monitoring**: Built-in WandB integration
5. **Safety**: Built-in safeguards for RL training



**TRL Training Progression Examples:**

 **SFT Code (What you used):**

    # 🎯 Stage 1: SFT (Current)
    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        formatting_func=formatting_func,
        data_collator=DataCollatorForCompletionOnlyLM(...)
    )
    trainer.train()
    

==================================================

 **Reward Model Code (Next step):**

    # 🏆 Stage 2: Reward Model Training
    from trl import RewardTrainer, RewardConfig

    # Dataset format: {"chosen": "good response", "rejected": "bad response"}
    trainer = RewardTrainer(
        model=model,
        train_dataset=preference_dataset,
        eval_dataset=eval_preference_dataset,
        args=RewardConfig(...)
    )
    trainer.train()
    

==================================================

 **RLHF/PPO Code (Advanced):**

    # 🔄 Stage 3: RLHF with PPO
    from trl import PPOTrainer, PPOConfig

    ppo_trainer = PPOTrainer(
        model=sft_model,
        ref_model=ref_model,  # Reference model (copy of SFT model)
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=PPOConfig(...)
    )

    # Training loop
    for batch in dataloader:
        query_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(query_tensors)
        rewards = reward_model(query_tensors, response_tensors)

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    

==================================================

 **DPO Code (Modern alternative):**

    # 🚀 Stage 4: DPO (Direct Preference Optimization)
    from trl import DPOTrainer, DPOConfig

    trainer = DPOTrainer(
        model=sft_model,
        ref_model=ref_model,
        train_dataset=preference_dataset,
        tokenizer=tokenizer,
        args=DPOConfig(beta=0.1, ...)  # beta controls preference strength
    )
    trainer.train()


 **Dataset Format Examples:**

 **SFT Dataset Format:**
{'conversations': [[{'from': 'human', 'value': 'What is AI?'}, {'from': 'gpt', 'value': 'AI is artificial intelligence...'}]]}

 **Reward Model Dataset Format:**
{'prompt': 'What is AI?', 'chosen': 'AI is artificial intelligence, a field of computer science...', 'rejected': 'AI is robots taking over the world!!!'}

 **PPO Dataset Format:**
{'query': 'What is AI?'}
   → PPO only needs prompts, no target responses!
