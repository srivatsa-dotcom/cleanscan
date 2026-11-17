# PlatformIO builds lib/cleanscan_proto itself (library dependency finder) and links it in.
idf_component_register(SRCS "main.cpp"
                       REQUIRES esp_wifi esp_netif esp_event nvs_flash esp_driver_uart esp_timer)
