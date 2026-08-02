using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using System.Windows.Media.Imaging;

using Emgu.CV;
using Emgu.CV.Util;

namespace SimpleDeHaze.Gui.Modern
{
    /// <summary>База для ViewModel: INotifyPropertyChanged + Set(ref field, value).</summary>
    public abstract class Obs : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;

        protected bool Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value)) return false;
            field = value;
            Raise(name);
            return true;
        }

        public void Raise([CallerMemberName] string? name = null)
            => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }

    /// <summary>Синхронная команда.</summary>
    public sealed class Cmd : ICommand
    {
        private readonly Action<object?> _exec;
        private readonly Func<object?, bool>? _can;

        public Cmd(Action<object?> exec, Func<object?, bool>? can = null) { _exec = exec; _can = can; }
        public Cmd(Action exec, Func<bool>? can = null) : this(_ => exec(), can == null ? null : _ => can()) { }

        public event EventHandler? CanExecuteChanged;
        public bool CanExecute(object? p) => _can?.Invoke(p) ?? true;
        public void Execute(object? p) => _exec(p);
        public void Refresh() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
    }

    /// <summary>Асинхронная команда с защитой от повторного запуска.</summary>
    public sealed class AsyncCmd : ICommand
    {
        private readonly Func<object?, Task> _exec;
        private readonly Func<bool>? _can;
        private bool _running;

        public AsyncCmd(Func<Task> exec, Func<bool>? can = null) { _exec = _ => exec(); _can = can; }
        public AsyncCmd(Func<object?, Task> exec, Func<bool>? can = null) { _exec = exec; _can = can; }

        public event EventHandler? CanExecuteChanged;
        public bool CanExecute(object? p) => !_running && (_can?.Invoke() ?? true);

        public async void Execute(object? p)
        {
            _running = true; Refresh();
            try { await _exec(p); }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show(ex.Message, "Ошибка выполнения",
                    System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Error);
            }
            finally { _running = false; Refresh(); }
        }

        public void Refresh() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
    }

    /// <summary>Маршалинг в UI-поток из фоновых задач (подбор, живое превью).</summary>
    public static class App
    {
        public static void Post(Action action)
        {
            var d = System.Windows.Application.Current?.Dispatcher;
            if (d == null || d.CheckAccess()) action();
            else d.BeginInvoke(action);
        }
    }

    public static class Img
    {
        /// <summary>Mat (8U BGR) -> BitmapSource через PNG в памяти. Freeze -> можно отдавать в UI-поток.</summary>
        public static BitmapSource ToBitmap(Mat mat)
        {
            using var buf = new VectorOfByte();
            CvInvoke.Imencode(".png", mat, buf);
            using var ms = new MemoryStream(buf.ToArray());
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.StreamSource = ms;
            bmp.EndInit();
            bmp.Freeze();
            return bmp;
        }
    }
}
