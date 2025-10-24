#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, strum::EnumIter)]
pub enum TurnDir {
    Clockwise,
    CounterClockwise,
    Double
}