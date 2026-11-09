import os
from datetime import datetime, timedelta


def changeFormat(s: str) -> str:
    s = s.replace('{', '{{{{')
    s = s.replace('}', '}}}}')
    print(s)
    return s


def unix_timestamp_to_time(timestamp, timezone_hours=8):
    """
    将 Unix 时间戳（秒）转换为具体时间
    :param timestamp: 整数，Unix 时间戳（如 1750000000）
    :param timezone_hours: 时区偏移小时数（默认 8，对应北京时间 UTC+8）
    :return: 字典，包含 UTC 时间、指定时区时间
    """
    utc_time = datetime.fromtimestamp(timestamp)
    local_time = utc_time + timedelta(hours=timezone_hours)

    result = {
        "timestamp": timestamp,
        "utc_time": utc_time.strftime("%Y-%m-%d %H:%M:%S"),
        "local_time": local_time.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": f"UTC+{timezone_hours}"
    }
    print(result)
    return result


def parse_datetime_to_timestamp(dt_str, timezone_hours=0):
    """
    将日期字符串转换为Unix时间戳

    Args:
        dt_str: 日期字符串，支持格式 "2022-04-01", "2022-04-01 12:00:00", "2022/04/01"
        timezone_hours: 时区偏移小时数，默认0(UTC)

    Returns:
        int: Unix时间戳

    Examples:
        >>> parse_datetime_to_timestamp("2022-04-01")
        1648780800
        >>> parse_datetime_to_timestamp("2022-04-01 12:00:00")
        1648824000
    """
    dt_str = dt_str.strip()

    formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]

    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Unsupported date format: {dt_str}. Supported formats: {formats}")

    dt = dt - timedelta(hours=timezone_hours)
    return int(dt.timestamp())


def parse_timestamp_to_datetime(timestamp, timezone_hours=0):
    """
    将Unix时间戳转换为日期时间字符串

    Args:
        timestamp: Unix时间戳（秒）
        timezone_hours: 时区偏移小时数，默认0(UTC)

    Returns:
        str: 格式化的日期时间字符串 "YYYY-MM-DD HH:MM:SS"
    """
    dt = datetime.fromtimestamp(timestamp)
    dt = dt + timedelta(hours=timezone_hours)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def backup_existing_file(path, timestamp=None):
    """Move an existing file to a timestamped backup path before rewriting it."""
    if not path or not os.path.exists(path):
        return None

    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    root, ext = os.path.splitext(path)
    backup_path = f"{root}.bak_{stamp}{ext}"
    counter = 1

    while os.path.exists(backup_path):
        backup_path = f"{root}.bak_{stamp}_{counter}{ext}"
        counter += 1

    os.replace(path, backup_path)
    return backup_path


def save_csv_with_backup(df, path, index=False):
    """Save a CSV while preserving any previous file at the same path."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    backup_path = backup_existing_file(path)
    df.to_csv(path, index=index)
    return backup_path, path


if __name__ == '__main__':
    # 测试示例
    s = """
    {
      tokenHourDatas(
        first: 10
        where: {periodStartUnix_lte: 1750000000, token: "0x0000000000085d4780b73119b644ae5ecd22b376"}
      ) {
        id
        low
        priceUSD
        periodStartUnix
        token {
          name
          id
          symbol
        }
      }
      tokens(first: 10) {
        symbol
        name
        id
      }
    }
    """
    s1 = changeFormat(s)
    unix_timestamp_to_time(1621821600)

    # 测试时间转换函数
    print("\n--- 测试 parse_datetime_to_timestamp ---")
    ts = parse_datetime_to_timestamp("2022-04-01")
    print(f"2022-04-01 -> {ts}")

    ts = parse_datetime_to_timestamp("2022-04-01 12:00:00")
    print(f"2022-04-01 12:00:00 -> {ts}")

    ts = parse_datetime_to_timestamp("2023-03-10")
    print(f"2023-03-10 -> {ts}")

    print("\n--- 测试 parse_timestamp_to_datetime ---")
    dt = parse_timestamp_to_datetime(1677436776)
    print(f"1648780800 -> {dt}")

    dt = parse_timestamp_to_datetime(1680278400)
    print(f"1648824000 -> {dt}")
