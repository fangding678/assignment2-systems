import logging
import socket
from datetime import datetime, timedelta
import torch
from torch.autograd.profiler import record_function
from torch_util import get_device
from cs336_basics.model import BasicsTransformerLM, softmax
# https://pytorch.org/blog/understanding-gpu-memory-1/


logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )

def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not exporting memory snapshot")
       return

   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   try:
       logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
   except Exception as e:
       logger.error(f"Failed to capture memory snapshot {e}")
       return

params_dic = {
    'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
    'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
    'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
    '2.7B': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32},
}
p_dic = params_dic['large']
d_model, d_ff, num_layers, num_heads = p_dic['d_model'], p_dic['d_ff'], p_dic['num_layers'], p_dic['num_heads']
vocab_size = 10000
batch_size = 4
context_length = 256
rope_theta = 10000

# Simple model example to demonstrate how to capture memory visuals.
def run_model1(num_iters=5, device="cuda:0"):
   model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(get_device())
   inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=get_device())
   labels = torch.rand_like(model(inputs))
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   loss_fn = torch.nn.CrossEntropyLoss()

   # Start recording memory snapshot history
   start_record_memory_history()

   for _ in range(num_iters):
       pred = model(inputs)
       loss_fn(pred, labels).backward()
       optimizer.step()
       optimizer.zero_grad(set_to_none=True)

   # Create the memory snapshot file
   export_memory_snapshot()

   # Stop recording memory snapshot history
   stop_record_memory_history()


def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   # Construct the trace file.
   prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")

def run_model2(num_iters=5, device="cuda:0"):
   model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(get_device())
   inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=get_device())
   labels = torch.rand_like(model(inputs))
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   loss_fn = torch.nn.CrossEntropyLoss()

   with torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA,
       ],
       schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
       record_shapes=True,
       profile_memory=True,
       with_stack=True,
       on_trace_ready=trace_handler,
   ) as prof:
       for _ in range(num_iters):
           prof.step()
           with record_function("## forward ##"):
               pred = model(inputs)

           with record_function("## backward ##"):
               loss_fn(pred, labels).backward()

           with record_function("## optimizer ##"):
               optimizer.step()
               optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # run_model1()
    run_model2()
    pass




