import json

llm_hosts = [
    "29.209.114.166",
    "29.209.113.113",
    "29.209.106.237",
    "29.209.114.56",
    "29.209.114.241",
    "29.209.112.177",
    "29.209.113.235"
]

t2vid_hosts = [
    "29.209.113.228",
    "29.209.105.60",
    "29.209.113.166",
    "29.209.113.176",
    "29.209.113.169",
    "29.209.112.74",
    "29.209.115.174",
    "29.209.113.156"
]

ports = [8200, 8201, 8202, 8203, 8204, 8205, 8206, 8207]

def create_instances(hosts, ports):
    instances = []
    for host in hosts:
        for port in ports:
            instances.append({
                "url": f"http://{host}:{port}",
                "hardware_name": "NVIDIA H20",
                "software_name": "sglang",
                "software_version": "0.5.5.post2"
            })
    return instances

config = {
    "dataset": "nkp37/OpenVid-1M",
    "llm_service": {
        "instances": create_instances(llm_hosts, ports)
    },
    "t2vid_service": {
        "instances": create_instances(t2vid_hosts, ports)
    },
    "predictor": {
        "url": "http://29.209.113.113:8100"
    },
    "execution": {
        "timeout": 300,
        "max_concurrent_requests": 100
    },
    "training_config": {
        "quantiles": [0.1, 0.5, 0.9]
    }
}

with open("experiments/03.Exp4.Text2Video/config_pipeline_example.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"Generated config with {len(config['llm_service']['instances'])} LLM instances and {len(config['t2vid_service']['instances'])} T2Vid instances.")
