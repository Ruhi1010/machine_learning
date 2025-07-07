import qrcode

# Ask for user input
data = input("Enter the data to encode in the QR code: ")

# Generate QR code
qr = qrcode.QRCode(
    version=1,  # size of the QR code: 1 is the smallest
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,  # size of each box in the QR code grid
    border=4,  # thickness of the border (minimum is 4)
)

qr.add_data(data)
qr.make(fit=True)

# Create an image
img = qr.make_image(fill_color="black", back_color="white")

# Save the image
img.save("user_qr_code.png")

print("QR code generated and saved as 'user_qr_code.png'")
