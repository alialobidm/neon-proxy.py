
resource "hcloud_server" "proxy" {
  name        = "proxy-${var.run_number}-${var.proxy_image_tag}"
  image       = data.hcloud_image.ci-image.id
  server_type = var.server_type
  location    = var.location
  ssh_keys = [
    data.hcloud_ssh_key.ci-ssh-key.id
  ]

  public_net {
    ipv4_enabled = true
    ipv6_enabled = false
  }

  network {
    network_id = data.hcloud_network.ci-network.id
  }

  provisioner "file" {
    source      = "../../../docker-compose/docker-compose-ci.yml"
    destination = "/tmp/docker-compose-ci.yml"

    connection {
        type        = "ssh"
        user        = "root"
        host        = hcloud_server.proxy.ipv4_address
        private_key = file("~/.ssh/ci-stands")
      }
  }

  provisioner "file" {
    content     = data.template_file.proxy_init.rendered
    destination = "/tmp/proxy_init.sh"

    connection {
    type        = "ssh"
    user        = "root"
    host        = hcloud_server.proxy.ipv4_address
    private_key = file("~/.ssh/ci-stands")
    }

  }


  provisioner "remote-exec" {
    inline = [
      "echo '${hcloud_server.solana.network.*.ip[0]}' > /tmp/solana_host",
      "chmod a+x /tmp/proxy_init.sh",
      "sudo /tmp/proxy_init.sh"
    ]

  connection {
    type        = "ssh"
    user        = "root"
    host        = hcloud_server.proxy.ipv4_address
    private_key = file("~/.ssh/ci-stands")
  }
  
  }

  labels = {
    environment = "ci"
    purpose    = "ci-oz-full-tests"
  }
  depends_on = [
    hcloud_server.solana
  ]
}

resource "hcloud_server" "solana" {
  name        = "solana-${var.run_number}-${replace(var.neon_evm_commit, "_", "-")}"
  image       = data.hcloud_image.ci-image.id
  server_type = var.server_type
  location    = var.location
  ssh_keys = [
    data.hcloud_ssh_key.ci-ssh-key.id
  ]

  provisioner "file" {
    source     = "../../../docker-compose/nginx.conf"
    destination = "/tmp/nginx.conf"

    connection {
        type        = "ssh"
        user        = "root"
        host        = hcloud_server.solana.ipv4_address
        private_key = file("~/.ssh/ci-stands")
      }
  }

   provisioner "file" {
    source      = "../../../docker-compose/docker-compose-ci.yml"
    destination = "/tmp/docker-compose-ci.yml"

    connection {
        type        = "ssh"
        user        = "root"
        host        = hcloud_server.solana.ipv4_address
        private_key = file("~/.ssh/ci-stands")
      }
  }

  public_net {
    ipv4_enabled = true
    ipv6_enabled = false
  }

  network {
    network_id = data.hcloud_network.ci-network.id
  }

  user_data = data.template_file.solana_init.rendered

  labels = {
    environment = "ci"
  }
}
