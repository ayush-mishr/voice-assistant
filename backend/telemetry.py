"""
OpenTelemetry Setup for AWS X-Ray & CloudWatch
==============================================
Initializes the tracer provider with the AWS X-Ray ID generator
and OTLP exporter to send traces to the local ADOT sidecar.
"""
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.extension.aws.trace import AwsXRayIdGenerator
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

logger = logging.getLogger("voice-server")

def init_telemetry():
    """Initialize OpenTelemetry globally with AWS X-Ray formatting."""
    
    # 1. Set the Service Name for X-Ray Service Map
    resource = Resource(attributes={
        SERVICE_NAME: "voice-assistant-backend"
    })

    # 2. Use AWS X-Ray ID Generator to ensure compatability
    provider = TracerProvider(
        id_generator=AwsXRayIdGenerator(),
        resource=resource
    )

    # 3. Add Exporter to ADOT local sidecar (default AWS port 4317)
    # The ADOT sidecar container will forward this to AWS X-Ray/CloudWatch.
    try:
        otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    except Exception as e:
        logger.warning(f"Could not initialize OTLP exporter: {e}")

    # OPTIONAL: Also export to console for local debugging visibility
    # provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    logger.info("[Telemetry] OpenTelemetry initialized with AWS X-Ray generator.")

def get_tracer(module_name: str):
    """Retrieve the global tracer instance."""
    return trace.get_tracer(module_name)
