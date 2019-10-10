import logging

lc_logger = logging.getLogger('dataset_logger')
lc_logger.setLevel(logging.DEBUG)
# Create handlers
f_handler = logging.FileHandler('dataset.log')
f_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                              "%Y-%m-%d %H:%M:%S")
f_handler.setFormatter(f_format)

# Add handlers to the logger
lc_logger.addHandler(f_handler)
