# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for working with tensorboard."""

from typing import Dict


def parse_summary_file(path: str) -> Dict[str, list]:
    """Parses 'Event' protobufs from a tensorboard summary file enabling programmatic manipulation of the results.

    Parameters
    ----------
    path: str
        The path to the event file

    Returns
    -------
    Dict[str, list]
        A "tall" dictionary mapping unique tuples of (key, step) to the corresponding recorded value

    """
    try:
        # is tf2 installed
        from tensorflow.compat.v1.train import summary_iterator
    except ImportError:
        try:
            from tensorflow.train import summary_iterator
        except ImportError:
            raise RuntimeError("tensorflow (1.x or 2.x) must be installed to parse summary files")

    keys = []
    vals = []
    time = []
    step = []
    for event in summary_iterator(path):
        if len(event.summary.value) == 0:
            continue

        for value in event.summary.value:
            keys.append(value.tag)
            step.append(event.step if getattr(event, "step", None) else 0)
            time.append(event.wall_time)

            try:
                # value.value is a oneof. we first determine which field, if any, is set and retrieve it. if no value
                # is set, it will raise a ValueError which we catch and replace value with a None object.
                vals.append(getattr(value, value.WhichOneof("value")))
            except ValueError:
                vals.append(None)

    return {"key": keys, "step": step, "time": time, "value": vals}
