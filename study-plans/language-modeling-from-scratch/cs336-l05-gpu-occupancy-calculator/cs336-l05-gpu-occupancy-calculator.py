import math
def gpu_occupancy(
    threads_per_block: int, registers_per_thread: int, shared_mem_per_block: int,
    max_threads_per_sm: int, max_warps_per_sm: int, max_blocks_per_sm: int,
    max_registers_per_sm: int, max_shared_mem_per_sm: int, warp_size: int = 32,
) -> dict:
    """
    Returns a dict: blocks_per_sm (int), resident_warps (int), occupancy (float).
    """
    output = {"blocks_per_sm":0,"resident_warps":0,"occupancy":0}
    
    
    warp_allocated = math.ceil(threads_per_block/warp_size)

    effective_threads = warp_allocated * warp_size
    register_use = effective_threads * registers_per_thread

    limits = [
        max_threads_per_sm // effective_threads,
        max_warps_per_sm // warp_allocated,
        max_blocks_per_sm,
    ]

    if registers_per_thread > 0:
        limits.append(max_registers_per_sm // register_use)

    if shared_mem_per_block > 0:
        limits.append(max_shared_mem_per_sm // shared_mem_per_block)

    blocks_per_sm = min(limits)
    resident_warps = blocks_per_sm * warp_allocated

    occupancy = resident_warps / max_warps_per_sm

    output["blocks_per_sm"] = blocks_per_sm
    output["resident_warps"] = resident_warps
    output["occupancy"] = occupancy
    
    return output
