#!/bin/bash
# Mount the navsim squashfs image and create symlinks in the navsim workspace.
# Usage (in your SLURM job script):
#   source /project/6061241/rdesc/extract_navsim.sh ~/navsim_workspace/navsim
# After sourcing, $OPENSCENE_DATA_ROOT points to the dataset root.

_extract_navsim() {
    local NAVSIM_ROOT="${1:?Usage: source extract_navsim.sh <navsim_devkit_path>}"
    local SQSH="/project/6061241/rdesc/navsim.sqsh"
    local SQSH_MOUNT="${SLURM_TMPDIR:?SLURM_TMPDIR not set}/.navsim_sqsh"
    local DATASET="$NAVSIM_ROOT/dataset"

    if [ ! -f "$SQSH" ]; then
        echo "ERROR: $SQSH not found. Run build_navsim_squashfs.sh first."
        return 1
    fi

    if [ ! -d "$NAVSIM_ROOT" ]; then
        echo "ERROR: $NAVSIM_ROOT does not exist."
        return 1
    fi

    mkdir -p "$SQSH_MOUNT" || return 1
    squashfuse "$SQSH" "$SQSH_MOUNT" || return 1

    rm -rf "$DATASET"
    mkdir -p "$DATASET" || return 1
    ln -sf "$SQSH_MOUNT/maps" "$DATASET/maps"
    ln -sf "$SQSH_MOUNT/navhard_two_stage" "$DATASET/navhard_two_stage"
    ln -sf "$SQSH_MOUNT/warmup_two_stage" "$DATASET/warmup_two_stage"

    # The squashfs has double-nested trainval dirs (navsim_logs/trainval/trainval/),
    # but the code expects pkl files directly under navsim_logs/trainval/.
    mkdir -p "$DATASET/navsim_logs" "$DATASET/sensor_blobs" || return 1
    if [ -d "$SQSH_MOUNT/navsim_logs/trainval/trainval" ]; then
        ln -sf "$SQSH_MOUNT/navsim_logs/trainval/trainval" "$DATASET/navsim_logs/trainval"
    else
        ln -sf "$SQSH_MOUNT/navsim_logs/trainval" "$DATASET/navsim_logs/trainval"
    fi
    if [ -d "$SQSH_MOUNT/sensor_blobs/trainval/trainval" ]; then
        ln -sf "$SQSH_MOUNT/sensor_blobs/trainval/trainval" "$DATASET/sensor_blobs/trainval"
    else
        ln -sf "$SQSH_MOUNT/sensor_blobs/trainval" "$DATASET/sensor_blobs/trainval"
    fi

    export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
    export NUPLAN_MAPS_ROOT="$DATASET/maps"
    export OPENSCENE_DATA_ROOT="$DATASET"
    export NAVSIM_EXP_ROOT="$NAVSIM_ROOT/exp"
    export NAVSIM_DEVKIT_ROOT="$NAVSIM_ROOT"

    mkdir -p "$NAVSIM_EXP_ROOT"

    echo "navsim dataset ready at: $OPENSCENE_DATA_ROOT"
    echo ""
    echo "  maps:                      $DATASET/maps"
    echo "  navsim_logs/trainval:      $DATASET/navsim_logs/trainval"
    echo "  sensor_blobs/trainval:     $DATASET/sensor_blobs/trainval"
    echo "  navhard_two_stage:         $DATASET/navhard_two_stage"
    echo "  warmup_two_stage:          $DATASET/warmup_two_stage"
}

_extract_navsim "$@"
unset -f _extract_navsim
