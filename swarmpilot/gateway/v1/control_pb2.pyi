from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class WatchStateRequest(_message.Message):
    __slots__ = ("gateway_id", "protocol_version")
    GATEWAY_ID_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    gateway_id: str
    protocol_version: int
    def __init__(self, gateway_id: _Optional[str] = ..., protocol_version: _Optional[int] = ...) -> None: ...

class State(_message.Message):
    __slots__ = ("version", "protocol_version", "model_id", "strategy", "proxy", "instances", "manifests", "predictor", "api_style_paths")
    class ManifestsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class ApiStylePathsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VERSION_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    PROXY_FIELD_NUMBER: _ClassVar[int]
    INSTANCES_FIELD_NUMBER: _ClassVar[int]
    MANIFESTS_FIELD_NUMBER: _ClassVar[int]
    PREDICTOR_FIELD_NUMBER: _ClassVar[int]
    API_STYLE_PATHS_FIELD_NUMBER: _ClassVar[int]
    version: int
    protocol_version: int
    model_id: str
    strategy: str
    proxy: ProxySettings
    instances: _containers.RepeatedCompositeFieldContainer[Instance]
    manifests: _containers.ScalarMap[str, str]
    predictor: PredictorSettings
    api_style_paths: _containers.ScalarMap[str, str]
    def __init__(self, version: _Optional[int] = ..., protocol_version: _Optional[int] = ..., model_id: _Optional[str] = ..., strategy: _Optional[str] = ..., proxy: _Optional[_Union[ProxySettings, _Mapping]] = ..., instances: _Optional[_Iterable[_Union[Instance, _Mapping]]] = ..., manifests: _Optional[_Mapping[str, str]] = ..., predictor: _Optional[_Union[PredictorSettings, _Mapping]] = ..., api_style_paths: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ProxySettings(_message.Message):
    __slots__ = ("timeout_s", "max_retries", "retry_delay_s", "http_timeout_s")
    TIMEOUT_S_FIELD_NUMBER: _ClassVar[int]
    MAX_RETRIES_FIELD_NUMBER: _ClassVar[int]
    RETRY_DELAY_S_FIELD_NUMBER: _ClassVar[int]
    HTTP_TIMEOUT_S_FIELD_NUMBER: _ClassVar[int]
    timeout_s: float
    max_retries: int
    retry_delay_s: float
    http_timeout_s: float
    def __init__(self, timeout_s: _Optional[float] = ..., max_retries: _Optional[int] = ..., retry_delay_s: _Optional[float] = ..., http_timeout_s: _Optional[float] = ...) -> None: ...

class Instance(_message.Message):
    __slots__ = ("instance_id", "endpoint", "model_id", "platform_info", "status", "max_in_flight", "servers", "extra_headers", "platform_key")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATUS_UNSPECIFIED: _ClassVar[Instance.Status]
        ACTIVE: _ClassVar[Instance.Status]
        DRAINING: _ClassVar[Instance.Status]
        UNHEALTHY: _ClassVar[Instance.Status]
        INITIALIZING: _ClassVar[Instance.Status]
        REMOVING: _ClassVar[Instance.Status]
    STATUS_UNSPECIFIED: Instance.Status
    ACTIVE: Instance.Status
    DRAINING: Instance.Status
    UNHEALTHY: Instance.Status
    INITIALIZING: Instance.Status
    REMOVING: Instance.Status
    class PlatformInfoEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class ExtraHeadersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_INFO_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MAX_IN_FLIGHT_FIELD_NUMBER: _ClassVar[int]
    SERVERS_FIELD_NUMBER: _ClassVar[int]
    EXTRA_HEADERS_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_KEY_FIELD_NUMBER: _ClassVar[int]
    instance_id: str
    endpoint: str
    model_id: str
    platform_info: _containers.ScalarMap[str, str]
    status: Instance.Status
    max_in_flight: int
    servers: int
    extra_headers: _containers.ScalarMap[str, str]
    platform_key: str
    def __init__(self, instance_id: _Optional[str] = ..., endpoint: _Optional[str] = ..., model_id: _Optional[str] = ..., platform_info: _Optional[_Mapping[str, str]] = ..., status: _Optional[_Union[Instance.Status, str]] = ..., max_in_flight: _Optional[int] = ..., servers: _Optional[int] = ..., extra_headers: _Optional[_Mapping[str, str]] = ..., platform_key: _Optional[str] = ...) -> None: ...

class PredictorSettings(_message.Message):
    __slots__ = ("storage_dir", "sketch_quantiles", "platforms", "preprocessor_rules", "experiment_normals")
    STORAGE_DIR_FIELD_NUMBER: _ClassVar[int]
    SKETCH_QUANTILES_FIELD_NUMBER: _ClassVar[int]
    PLATFORMS_FIELD_NUMBER: _ClassVar[int]
    PREPROCESSOR_RULES_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_NORMALS_FIELD_NUMBER: _ClassVar[int]
    storage_dir: str
    sketch_quantiles: _containers.RepeatedScalarFieldContainer[float]
    platforms: _containers.RepeatedCompositeFieldContainer[Platform]
    preprocessor_rules: bool
    experiment_normals: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, storage_dir: _Optional[str] = ..., sketch_quantiles: _Optional[_Iterable[float]] = ..., platforms: _Optional[_Iterable[_Union[Platform, _Mapping]]] = ..., preprocessor_rules: _Optional[bool] = ..., experiment_normals: _Optional[_Iterable[float]] = ...) -> None: ...

class Platform(_message.Message):
    __slots__ = ("platform_key", "quantile_model_key", "expect_error_model_key", "hardware_features", "experiment")
    class HardwareFeaturesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    PLATFORM_KEY_FIELD_NUMBER: _ClassVar[int]
    QUANTILE_MODEL_KEY_FIELD_NUMBER: _ClassVar[int]
    EXPECT_ERROR_MODEL_KEY_FIELD_NUMBER: _ClassVar[int]
    HARDWARE_FEATURES_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_FIELD_NUMBER: _ClassVar[int]
    platform_key: str
    quantile_model_key: str
    expect_error_model_key: str
    hardware_features: _containers.ScalarMap[str, float]
    experiment: bool
    def __init__(self, platform_key: _Optional[str] = ..., quantile_model_key: _Optional[str] = ..., expect_error_model_key: _Optional[str] = ..., hardware_features: _Optional[_Mapping[str, float]] = ..., experiment: _Optional[bool] = ...) -> None: ...

class CompletionBatch(_message.Message):
    __slots__ = ("completions", "arrivals", "calls_from")
    class CallsFromEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    COMPLETIONS_FIELD_NUMBER: _ClassVar[int]
    ARRIVALS_FIELD_NUMBER: _ClassVar[int]
    CALLS_FROM_FIELD_NUMBER: _ClassVar[int]
    completions: _containers.RepeatedCompositeFieldContainer[Completion]
    arrivals: int
    calls_from: _containers.ScalarMap[str, int]
    def __init__(self, completions: _Optional[_Iterable[_Union[Completion, _Mapping]]] = ..., arrivals: _Optional[int] = ..., calls_from: _Optional[_Mapping[str, int]] = ...) -> None: ...

class Completion(_message.Message):
    __slots__ = ("task_id", "instance_id", "model_id", "path", "features", "predicted_time_ms", "predicted_error_margin_ms", "predicted_quantiles", "succeeded", "http_status", "execution_time_ms", "error")
    class FeaturesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class PredictedQuantilesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_ERROR_MARGIN_MS_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_QUANTILES_FIELD_NUMBER: _ClassVar[int]
    SUCCEEDED_FIELD_NUMBER: _ClassVar[int]
    HTTP_STATUS_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    instance_id: str
    model_id: str
    path: str
    features: _containers.ScalarMap[str, float]
    predicted_time_ms: float
    predicted_error_margin_ms: float
    predicted_quantiles: _containers.ScalarMap[str, float]
    succeeded: bool
    http_status: int
    execution_time_ms: float
    error: str
    def __init__(self, task_id: _Optional[str] = ..., instance_id: _Optional[str] = ..., model_id: _Optional[str] = ..., path: _Optional[str] = ..., features: _Optional[_Mapping[str, float]] = ..., predicted_time_ms: _Optional[float] = ..., predicted_error_margin_ms: _Optional[float] = ..., predicted_quantiles: _Optional[_Mapping[str, float]] = ..., succeeded: _Optional[bool] = ..., http_status: _Optional[int] = ..., execution_time_ms: _Optional[float] = ..., error: _Optional[str] = ...) -> None: ...

class UnreachableReport(_message.Message):
    __slots__ = ("instance_id", "error")
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    instance_id: str
    error: str
    def __init__(self, instance_id: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class ReportAck(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
