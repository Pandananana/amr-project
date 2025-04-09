## Play ROSBAG

Open terminal and run `roscore` then you can run `rosbag play -l rosbag.bag`


## Extract Oculus data from ROSBAG

Modify the following functions in `bps_oculus/core.py` 

```python
def unpack_ping(item_payload: bytes) -> Union[oculus.OculusSimplePingResult, oculus.OculusSimplePingResult2, None]:
    """
    Given a bytes object, attempt to unpack a SimplePingResult (v1 or v2) structure.
    :param item_payload: the item_message's payload as a bytes object
    :return: either an oculus.OculusSimplePingResult or oculus.OculusSimplePingResult2 struct.
    """
    # Extract the header
    head = oculus.OculusMessageHeader.unpack(item_payload[:oculus.OculusMessageHeader.size()])
    if head.oculusId != oculus.OCULUS_CHECK_ID:
        raise ValueError(f"OculusID not {0xf453}, got {head.oculusId}")

    if head.msgId == oculus.OculusMessageType.messageSimplePingResult:
        ver = head.msgVersion
        if ver == 2:
            return oculus.OculusSimplePingResult2.unpack(item_payload[:oculus.OculusSimplePingResult2.size()])
        elif ver == 1:
            return oculus.OculusSimplePingResult.unpack(item_payload[:oculus.OculusSimplePingResult.size()])
        else:
            raise ValueError(f"Simple ping result version can only be 1 or 2. Got {ver}")
    elif head.msgId == oculus.OculusMessageType.messagePingResult:
        return oculus.OculusReturnFireMessage.unpack(item_payload[:oculus.OculusReturnFireMessage.size()])
    elif head.msgId == oculus.OculusMessageType.messageDummy:
        return None
    else:
        return None


def unpack_data_entry(entry: bytes) -> Union[Tuple[Union[oculus.OculusSimplePingResult, oculus.OculusSimplePingResult2], oculus.OculusPolarImage, bytes], None]:
    ping_result = unpack_ping(entry)
    if ping_result is None:
        return None, None, None
    polar_image_data = parse_polar_image(entry, ping_result)
    ping_result = filter_gain_result(ping_result)
    new_buffer = ping_result.pack() + entry[
                                      ping_result.size():ping_result.imageOffset] + polar_image_data.polar_image.tobytes()
    return ping_result, polar_image_data, new_buffer
```


### Dependencies

pip install rosbags


