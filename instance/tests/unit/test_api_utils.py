from src.server import construct_websocket_url

def test_construct_websocket_url():
  endpoint = "http://1.1.1.1:8000"
  
  ws_url = construct_websocket_url(endpoint)
  
  assert ws_url == "ws://1.1.1.1:8001/instance/ws"
  
def test_construct_websocket_url_no_port():
  endpoint = "http://1.1.1.1"
  
  ws_url = construct_websocket_url(endpoint)
  
  assert ws_url == "ws://1.1.1.1:8001/instance/ws"