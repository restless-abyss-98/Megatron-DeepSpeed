SCRIPT_DIR=/eagle/datascience/eku/MDS-VIT-CLEAN/examples_deepspeed/sequence_parallel ## TODO: CHANGE THIS
LOG_DIR=$SCRIPT_DIR/custom_log
ds_script=$SCRIPT_DIR/ds_pretrain_gpt_1.3B_seq_parallel_32k.sh

mkdir -p $LOG_DIR

export train_iter=10
export drop_last_batch_with_GBS=1   ## reconcile the data order between DP and MP
# export PROFILE=1                  ## Turn on Torch profiler
# export TPSP=1                     ## Turn on LM's Sequence Parallelism
MP=4                                ## Model parallelism degree

## USP (Working)
sp_size=$MP  zero_stage=3                             bash $ds_script  ## Deepspeed's Ulysses
# sp_size=$MP  zero_stage=2  USP_ulysses=1              bash $ds_script 
# sp_size=$MP  zero_stage=2  USP_ring=1                 bash $ds_script 
# sp_size=$MP  zero_stage=2  USP_hybrid=1  MODEL=SMALL  bash $ds_script 

## USP (Not Working)
## USP_hybrid goes OOM early
# sp_size=$MP USP_hybrid=1     bash $ds_script |& tee $LOG_DIR/ex_ds4.log 
# [rank0]: torch.distributed.DistBackendError: NCCL error in: /soft/applications/conda/2024-04-29/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1970, unhandled cuda error (run with NCCL_DEBUG=INFO for details), NCCL version 2.20.5
# [rank0]: ncclUnhandledCudaError: Call to CUDA function failed.
# [rank0]: Last error:
# [rank0]: Failed to CUDA calloc async 4 bytes

## USP breaks with zero3
# sp_size=$MP zero_stage=3 USP_ulysses=1 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_ZERO3.log bash $ds_script |& tee $LOG_DIR/ex_ds4.log
# [rank0]: torch.jit.frontend.NotSupportedError: Compiled functions can't take variable number of arguments or use keyword-only arguments with defaults:
# [rank0]:   File "/home/eku/venv/stable_ds15.1/lib/python3.11/site-packages/deepspeed/runtime/zero/partition_parameters.py", line 237
# [rank0]:     def wrapped_fn(*args, **kwargs) -> Tensor:
# [rank0]:                            ~~~~~~~ <--- HERE
# [rank0]:         if kwargs.get("device", None) is None:
# [rank0]:             kwargs['device'] = torch.device(get_accelerator().device_name(os.environ["LOCAL_RANK"]))









## MISC ##
## Zero-DP
# sp_size=1 zero_stage=0 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_DP4.log bash $ds_script |& tee $LOG_DIR/ex_ds1.log
# sp_size=1 zero_stage=1 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_ZERO1.log bash $ds_script |& tee $LOG_DIR/ex_ds2.log
# sp_size=1 zero_stage=2 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_ZERO2.log bash $ds_script |& tee $LOG_DIR/ex_ds3.log
# sp_size=1 zero_stage=2 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_ZERO3.log bash $ds_script |& tee $LOG_DIR/ex_ds4.log

## parallelisms
# sp_size=$MP zero_stage=3 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_SP4.log bash $ds_script |& tee $LOG_DIR/ex_ds5.log
# tp_size=$MP zero_stage=2 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_TP4.log bash $ds_script |& tee $LOG_DIR/ex_ds6.log
# tp_size=$MP zero_stage=3 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_TP4.log bash $ds_script |& tee $LOG_DIR/ex_ds6.log
# pp_size=$MP zero_stage=0 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_PP4.log bash $ds_script |& tee $LOG_DIR/ex_ds7.log
# SIZE=1 zero_stage=0 DATA_PATH_LOG=$LOG_DIR/consumed_tokens_DP1.log bash $ds_script |& tee $LOG_DIR/ex_ds8.log ## Only works on one-node