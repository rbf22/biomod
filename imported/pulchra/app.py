import argparse
import asyncio
import json
import logging
import os
from dotenv import load_dotenv

from mev_share_py.client import MevShareClient
from mev_share_py.event_stream import EventStream, EventType

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    """
    This is the main function that runs the MEV-share client.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='MEV-share client')
    parser.add_argument('--network', type=str, default='goerli', choices=['goerli', 'mainnet'], help='Network to connect to')
    args = parser.parse_args()

    # Load common config file
    with open('config/common.json', 'r') as f:
        common_config = json.load(f)

    # Load networks config file
    with open('config/networks.json', 'r') as f:
        networks_config = json.load(f)

    # Get network-specific config
    network_config = networks_config[args.network]

    # Merge configs
    config = {**common_config, **network_config}

    # Get environment variables
    execution_node_url = os.environ.get('EXECUTION_NODE_URL', config.get('execution_node_url'))
    beacon_node_url = os.environ.get('BEACON_NODE_URL', config.get('beacon_node_url'))
    builder_urls = os.environ.get('BUILDER_URLS', config.get('builder_urls'))
    redis_url = os.environ.get('REDIS_URL', config.get('redis_url'))
    port = os.environ.get('PORT', config.get('port'))
    network = os.environ.get('NETWORK', config.get('network'))
    version = os.environ.get('VERSION', config.get('version'))

    # Create event stream
    event_stream = EventStream(
        event_type=EventType.Bundle,
        network=network,
        execution_node_url=execution_node_url,
        beacon_node_url=beacon_node_url,
        redis_url=redis_url,
        port=port
    )

    # Create MEV-share client
    client = MevShareClient(
        event_stream=event_stream,
        builder_urls=builder_urls,
        version=version
    )

    # Run the client
    await client.run()


if __name__ == '__main__':
    asyncio.run(main())
