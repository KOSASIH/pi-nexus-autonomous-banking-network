import RPi.GPIO as GPIO


def setup_i2c(sda_pin, scl_pin):
    """
    Sets up the I2C communication protocol using the specified SDA and SCL pins.
    """
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(sda_pin, GPIO.OUT)
    GPIO.setup(scl_pin, GPIO.OUT)
    GPIO.output(sda_pin, GPIO.HIGH)
    GPIO.output(scl_pin, GPIO.HIGH)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(sda_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(scl_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def setup_spi(sclk_pin, mosi_pin, miso_pin):
    """
    Sets up the SPI communication protocol using the specified SCLK, MOSI, and MISO pins.
    """
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(sclk_pin, GPIO.OUT)
    GPIO.setup(mosi_pin, GPIO.OUT)
    GPIO.setup(miso_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def cleanup():
    """
    Cleans up the GPIO pins and resets the GPIO library.
    """
    GPIO.cleanup()
