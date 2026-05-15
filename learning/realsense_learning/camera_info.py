import pyrealsense2 as rs


def main():
    ctx = rs.context()
    if len(ctx.devices) > 0:
        for d in ctx.devices:
            print(f'Found device: {d.get_info(rs.camera_info.name)}, SN: {d.get_info(rs.camera_info.serial_number)}')
    else:
        print("No Intel Device connected")


if __name__ == "__main__":
    main()
