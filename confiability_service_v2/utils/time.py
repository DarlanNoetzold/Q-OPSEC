"""
Time Utilities - Funções para manipulação de timestamps e cálculos temporais
"""
from datetime import datetime, timezone
from typing import Optional


def now_utc() -> datetime:
    """
    Retorna datetime atual em UTC

    Returns:
        datetime em UTC
    """
    return datetime.now(timezone.utc)


def utc_now() -> datetime:
    """
    Alias para now_utc() - compatibilidade

    Returns:
        datetime em UTC
    """
    return now_utc()


def parse_iso(timestamp: any) -> Optional[datetime]:
    """
    Parse de timestamp em diversos formatos

    Args:
        timestamp: String ISO, datetime, ou timestamp Unix

    Returns:
        datetime ou None se inválido
    """
    if timestamp is None:
        return None

    # Já é datetime
    if isinstance(timestamp, datetime):
        # Garante que tem timezone
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp

    # String ISO
    if isinstance(timestamp, str):
        try:
            # Tenta parse ISO 8601
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, AttributeError):
            return None

    # Unix timestamp (int ou float)
    if isinstance(timestamp, (int, float)):
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (ValueError, OSError):
            return None

    return None


def parse_timestamp(timestamp: any) -> Optional[datetime]:
    """
    Alias para parse_iso() - compatibilidade

    Args:
        timestamp: String ISO, datetime, ou timestamp Unix

    Returns:
        datetime ou None se inválido
    """
    return parse_iso(timestamp)


def get_age_seconds(timestamp: any) -> float:
    """
    Calcula idade em segundos desde o timestamp

    Args:
        timestamp: Timestamp a calcular idade

    Returns:
        Idade em segundos (0.0 se inválido)
    """
    dt = parse_iso(timestamp)
    if dt is None:
        return 0.0

    now = now_utc()
    delta = now - dt
    return max(0.0, delta.total_seconds())


def get_age_hours(timestamp: any) -> float:
    """
    Calcula idade em horas desde o timestamp

    Args:
        timestamp: Timestamp a calcular idade

    Returns:
        Idade em horas (0.0 se inválido)
    """
    return get_age_seconds(timestamp) / 3600.0


def get_age_days(timestamp: any) -> float:
    """
    Calcula idade em dias desde o timestamp

    Args:
        timestamp: Timestamp a calcular idade

    Returns:
        Idade em dias (0.0 se inválido)
    """
    return get_age_hours(timestamp) / 24.0


def delta_seconds(timestamp1: any, timestamp2: any) -> float:
    """
    Calcula diferença em segundos entre dois timestamps

    Args:
        timestamp1: Primeiro timestamp
        timestamp2: Segundo timestamp

    Returns:
        Diferença em segundos (absoluta)
    """
    dt1 = parse_iso(timestamp1)
    dt2 = parse_iso(timestamp2)

    if dt1 is None or dt2 is None:
        return 0.0

    delta = abs((dt1 - dt2).total_seconds())
    return delta


def time_diff_seconds(timestamp1: any, timestamp2: any) -> float:
    """
    Alias para delta_seconds() - compatibilidade

    Args:
        timestamp1: Primeiro timestamp
        timestamp2: Segundo timestamp

    Returns:
        Diferença em segundos (absoluta)
    """
    return delta_seconds(timestamp1, timestamp2)


def is_expired(timestamp: any, max_age_hours: float) -> bool:
    """
    Verifica se timestamp expirou baseado em idade máxima

    Args:
        timestamp: Timestamp a verificar
        max_age_hours: Idade máxima em horas

    Returns:
        True se expirado
    """
    age = get_age_hours(timestamp)
    return age > max_age_hours


def format_iso(dt: datetime) -> str:
    """
    Formata datetime para ISO 8601

    Args:
        dt: datetime a formatar

    Returns:
        String ISO 8601
    """
    if dt is None:
        return ""

    # Garante UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.isoformat()
