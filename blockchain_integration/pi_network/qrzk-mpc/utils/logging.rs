// utils/logging.rs: Logging utility implementation

use log::{Level, LevelFilter, Log, Metadata, Record};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

static LOGGER_INITIALIZED: AtomicBool = AtomicBool::new(false);

pub struct Logger {
    level: Level,
}

impl Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
            let level = record.level();
            let target = record.target();
            let message = record.args();

            let output = format!(
                "[{} {} {}] {}",
                timestamp,
                level.to_string().to_uppercase(),
                target,
                message
            );

            let mut stdout = io::stdout();
            stdout.write_all(output.as_bytes()).unwrap();
            stdout.write_all(b"\n").unwrap();
            stdout.flush().unwrap();
        }
    }

    fn flush(&self) {}
}

pub fn init_logger(level: Level) {
    if LOGGER_INITIALIZED.swap(true, Ordering::SeqCst) {
        return;
    }

    log::set_boxed_logger(Box::new(Logger { level })).unwrap();
    log::set_max_level(level.to_level_filter());
}

pub fn debug<T: std::fmt::Debug>(message: T) {
    log::debug!("{:?}", message);
}

pub fn info<T: std::fmt::Display>(message: T) {
    log::info!("{}", message);
}

pub fn warn<T: std::fmt::Display>(message: T) {
    log::warn!("{}", message);
}

pub fn error<T: std::fmt::Display>(message: T) {
    log::error!("{}", message);
}

pub fn critical<T: std::fmt::Display>(message: T) {
    log::error!("{}", message);
}
