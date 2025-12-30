import sshtunnel
from sshtunnel import SSHTunnelForwarder, BaseSSHTunnelForwarderError

def create_ssh_tunnel(ssh_server_ip, ssh_username, ssh_password,
                      remote_port, local_port=5000):
    """
    Creates and starts an SSH tunnel to the remote server.

    Args:
        ssh_server_ip (str): The SSH server's IP address.
        ssh_username (str): The SSH username.
        ssh_password (str): The SSH password.
        remote_port (int): The remote port to bind on the server side.
        local_port (int, optional): The local port to bind the tunnel (default is 3000).

    Returns:
        SSHTunnelForwarder: The established SSH tunnel. Remember to stop() it when no longer needed.
    """
    try:
        tunnel = SSHTunnelForwarder(
            (ssh_server_ip, 22),
            ssh_username=ssh_username,
            ssh_password=ssh_password,
            allow_agent=False,
            remote_bind_address=('127.0.0.1', remote_port),
            local_bind_address=('127.0.0.1', local_port)
        )
        tunnel.start()
        print("[CLIENT] SSH tunnel established on local port:", tunnel.local_bind_port)
        return tunnel

    except BaseSSHTunnelForwarderError as e:
        print("[CLIENT] Failed to start SSH tunnel:", str(e))
        return None
    return tunnel