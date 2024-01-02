mkdir /home/aanwar/le000288/Efficient_representation_inference/src/msiout
sbatch clm0wikitext-2v1_llama-2-7b_clm_1_pqstructlocal+h+2+9999+-1+max_inter_somemethods-3_o-proj+down-proj.pbs --wait
mkdir /home/aanwar/le000288/Efficient_representation_inference/src/msiout
sbatch clm0wikitext-2v1_llama-2-7b_clm_1_pqstructlocal+h+2+9999+-1+max_inter_somemethods-3_default.pbs --wait
mkdir /home/aanwar/le000288/Efficient_representation_inference/src/msiout
sbatch clm0wikitext-2v1_opt-1.3b_clm_1_pqstructlocal+h+2+9999+-1+max_inter_somemethods-3_out-proj+fc2.pbs --wait
mkdir /home/aanwar/le000288/Efficient_representation_inference/src/msiout
sbatch clm0wikitext-2v1_opt-1.3b_clm_1_pqstructlocal+h+2+9999+-1+max_inter_somemethods-3_default.pbs --wait
